"""Reach control: plan latent actions toward a goal frame with the world model and turn each plan into robot commands with the IDM."""
import numpy as np
import torch
from torch import nn

from src.config import CHUNK, SIM_STRIDE
from src.features.dino import encode, load_model
from src.datasets.sim import PLAY, SimChunkDataset
from src.models.idm import InverseDynamics
from src.models.latent_action import LatentActionEncoder, encode_chunk
from src.models.world_model import WorldModel
from src.planning.cem import cem, fit_prior, token_distance
from src.sim import scenes

DEVICE = "cuda"
X, Y, Z = (-0.12, 0.12), (-0.22, 0.22), (0.86, 1.08)  # target ranges above the table in metres: the play-data workspace, kept clear of the objects
REACH = (0.15, 0.25)  # allowed start-to-target distance in metres
SUCCESS = 0.05  # a reach counts as done within this distance (m)


def load_models(cotrain, idm_checkpoint):
    """Load the co-trained encoder and world model and the inverse dynamics model, all in eval mode."""
    model = nn.ModuleDict({"encoder": LatentActionEncoder(), "world_model": WorldModel()}).to(DEVICE).eval()
    model.load_state_dict(torch.load(cotrain, map_location=DEVICE)["model"])
    saved = torch.load(idm_checkpoint, map_location=DEVICE)
    a = saved["args"]
    idm = InverseDynamics(a["d_model"], a["dropout"], a["layers"], a["head_hidden"], a.get("gripper", False)).to(DEVICE).eval()
    idm.load_state_dict(saved["model"])
    return model, idm


def dino_features():
    """Return a function mapping RGB frames (N, H, W, 3) to DINO patch features (N, tokens, 384) on the GPU."""
    dino, mean, std = load_model(DEVICE)
    return lambda frames: torch.from_numpy(encode(dino, mean, std, frames, DEVICE)).float().to(DEVICE)


def play_prior(model):
    """Fit the latent prior (mean and std per latent number) to the co-trained encoder's latents on 50 play episodes."""
    with torch.no_grad():
        z = torch.cat([encode_chunk(model["encoder"], f)[0].flatten(0, 1) for f, *_ in SimChunkDataset("train", DEVICE, (PLAY,), 50).batches(256, shuffle=False)])
    return fit_prior(z)


def plan_commands(current, goal, weights, model, idm, prior, weighted=False, metric="l2", hold=False):
    """Return the IDM's position commands (SIM_STRIDE, 3) for the first predicted frame of a latent plan: CEM toward the goal; `weighted` scores all CHUNK predicted frames with weights rising linearly so the last counts most, otherwise only the best-matching frame counts; `metric` is the per-token distance, "l2" or "cosine"; `hold` returns zero commands when the current frame already matches the goal at least as well as the best plan (use it with the default best-frame cost and "l2")."""
    steps = torch.arange(1, CHUNK + 1, device=DEVICE) / (CHUNK * (CHUNK + 1) / 2) if weighted else None
    with torch.no_grad(), torch.autocast(DEVICE, torch.bfloat16):
        plan, best = cem(model["world_model"], current, goal, prior, weights=weights, step_weights=steps, metric=metric)
        if hold and (token_distance(current, goal, metric) * weights).sum().item() <= best:
            return np.zeros((SIM_STRIDE, idm.outputs), dtype=np.float32)  # staying put is no worse than any plan
        predicted = model["world_model"](current, plan[None])[:, 0]  # the first predicted frame
        return idm(current, predicted).float()[0].clamp(-1, 1).cpu().numpy()


def grasp_and_lift(rec, half, shape):
    """Grasp the object with the scripted controller and lift it to hover height, then clear the recorder so it only holds what happens next; return the object's resting height."""
    obj = rec.obs["cube_pos"].copy()
    at = lambda z: np.array([obj[0], obj[1], z])
    kw = dict(mode="normal", half=half, shape=shape)
    rec.move(at(obj[2] + scenes.HOVER), -1, **kw)
    rec.move(at(obj[2] + 0.005), -1, **kw)
    rec.move(at(obj[2] + 0.005), 1, hold=12)
    rec.move(at(obj[2] + scenes.HOVER), 1)
    rec.data = {key: [] for key in rec.data}
    rec.snapshot()
    return obj[2]


def make_scene(seed, gripper="default", carry=False):
    """Build the seeded clutter scene and the given robosuite gripper; reach scenes hold a cube, carry scenes a cube, cuboid or cylinder in an unseen colour that the gripper starts out holding. Return the environment, a reset function giving a fresh recorder at the same start, and the scene's random stream."""
    rng = np.random.default_rng(seed)
    shape, colour = (scenes.SHAPES[seed % 3], scenes.TEST_COLOURS[seed % 4]) if carry else ("cube", scenes.TEST_COLOURS[0])
    half, _ = scenes.sample_object(shape, rng)
    env = scenes.make_env(shape, half, colour, scenes.sample_distractors(rng), gripper)

    def reset():
        """Reseed and return a fresh recorder at the scene's start."""
        np.random.seed(seed)
        rec = scenes.Recorder(env, env.reset())
        if carry:
            rec.rest_z = grasp_and_lift(rec, half, shape)
        return rec

    return env, reset, rng


def sample_target(start, rng, direction, avoid, floor):
    """Draw a target position 15-25 cm from the start (raising an error if none is free), inside the workspace, at least 8 cm (horizontally) from every object in `avoid` and above `floor`, whose horizontal direction from the start falls in the `direction` bin (k, n): the n equal bins of the forward +-90 degree arc, the only directions the workspace allows from the start pose."""
    k, n = direction
    low, width = -np.pi / 2, np.pi / n
    for _ in range(20000):
        p = np.array([rng.uniform(*X), rng.uniform(*Y), rng.uniform(max(Z[0], floor), Z[1])])
        angle = np.arctan2(p[1] - start[1], p[0] - start[0])
        if REACH[0] <= np.linalg.norm(p - start) <= REACH[1] and low + k * width <= angle < low + (k + 1) * width and all(np.linalg.norm(p[:2] - o[:2]) > 0.08 for o in avoid):
            return p
    raise RuntimeError(f"no free target in direction bin {direction}")


def in_view(mask, margin=8):
    """Return whether the mask is large enough and clear of the image border."""
    rows, cols = np.nonzero(mask)
    return len(rows) > 300 and rows.min() >= margin and cols.min() >= margin and rows.max() < mask.shape[0] - margin and cols.max() < mask.shape[1] - margin


def finger_closure(qpos, q_open, q_goal):
    """Return how far the fingers have moved from fully open (0) to the goal's finger positions (1); 1 when the goal's gripper is itself open."""
    span = np.linalg.norm(q_open - q_goal)
    return 1.0 if span < 1e-3 else float(1 - np.linalg.norm(qpos - q_goal) / span)


def run(reset, goal, goal_pos, features, model, idm, prior, budget, execute=SIM_STRIDE, weighted=False, metric="l2", fingers=None, hold=False, hold_grip=-1.0):
    """Replan `budget` times with the CEM planner, executing the first `execute` of the IDM's SIM_STRIDE commands each time, and stop once the end effector is within SUCCESS of the goal (and, given `fingers` = (open, goal) finger positions, the fingers are 90% of the way to the goal's); an IDM without a gripper output holds the gripper at `hold_grip` (-1 open, 1 closed); return the recorder and the distance to the goal before each replan and at the end."""
    rec = reset()
    frame = lambda: features(rec.obs["agentview_image"][::-1][None].copy())
    change = (goal - frame()[0]).pow(2).sum(-1)
    weights = change / change.sum()  # focus the planner on the patches that differ between start and goal
    distance = lambda: float(np.linalg.norm(rec.obs["robot0_eef_pos"] - goal_pos))
    distances = []
    for _ in range(budget):
        distances.append(distance())
        if distances[-1] < SUCCESS and (fingers is None or finger_closure(rec.obs["robot0_gripper_qpos"], *fingers) > 0.9):
            return rec, distances
        for command in plan_commands(frame(), goal, weights, model, idm, prior, weighted, metric, hold)[:execute]:
            grip = 1.0 if len(command) == 4 and command[3] > 0 else hold_grip  # the IDM's gripper command if it predicts one, otherwise the held value
            rec.step(np.concatenate([command[:3], np.zeros(3), [grip]]))  # no rotation
    distances.append(distance())
    return rec, distances


def make_goal(seed, features, gripper="default", goal_grip=-1, carry=False, direction=(0, 5)):
    """Build the seeded scene and render its goal by scripted motion with the gripper command `goal_grip` (-1 open, 1 closed); return a dict with the environment, its reset function, the goal's DINO features, end-effector position, RGB frame, gripper mask, whether a carried cube was actually held at the start, the finger joint positions of a fully open gripper and in the goal, the start-to-goal distance and the scripted arm's step count."""
    env, reset, rng = make_scene(seed, gripper, carry)
    start = reset()
    start_pos = start.obs["robot0_eef_pos"].copy()
    held = bool(carry and start.obs["cube_pos"][2] > start.rest_z + 0.05)  # the object came up with the gripper
    avoid = env.distractor_pos()
    floor = start.rest_z + 0.07 if carry else 0.0  # a carried object must clear the table
    shown = ("gripper0_", "cube_main") if carry else ("gripper0_",)
    k, n = direction
    bins = sorted(range(n), key=lambda b: abs(b - k))  # the assigned bin first, then its neighbours when clutter blocks it
    strict = not carry  # reach goals must be reachable and fully in view; carry goals prefer it but accept the last draw
    oracle = None
    for attempt in range(30 if strict else 8):
        clutter = avoid if strict or attempt < 4 else ()
        try:
            target = sample_target(start_pos, rng, (bins[attempt // 6 if strict else attempt % 4], n), clutter, floor)
        except RuntimeError:
            continue
        oracle = reset()
        oracle.move(target, goal_grip)  # scripted motion renders the goal frame
        mask, reached = scenes.body_mask(env, shown), oracle.obs["robot0_eef_pos"]
        if in_view(mask) and np.linalg.norm(reached - target) < 0.015:
            break
    else:
        if strict or oracle is None:
            raise RuntimeError(f"seed {seed}: no visible, reachable goal near direction bin {direction}")
    pos, rgb = reached.copy(), oracle.data["rgb"][-1]
    opened = reset()
    opened.wait(-1, 10)  # a gripper commanded open, to measure closure against
    return {"env": env, "reset": reset, "goal": features(rgb[None])[0], "pos": pos, "rgb": rgb, "mask": mask, "held": held, "qpos_open": np.array(opened.obs["robot0_gripper_qpos"]), "qpos_goal": np.array(oracle.data["gripper_qpos"][-1]), "start_distance": float(np.linalg.norm(pos - start_pos)), "scripted_steps": len(oracle.data["action"])}
