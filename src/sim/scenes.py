"""Generate scripted Panda pick-and-place demonstrations (including drops and missed grasps) of a cube, cuboid or cylinder among clutter, saving actions and gripper state per episode."""
import argparse
import csv
from pathlib import Path

import h5py
import imageio
import numpy as np
from robosuite import load_controller_config
from robosuite.environments.manipulation.lift import Lift
from robosuite.environments.manipulation.single_arm_env import SingleArmEnv
from robosuite.models.arenas import TableArena
from robosuite.models.objects import BallObject, BoxObject, CylinderObject
from robosuite.models.tasks import ManipulationTask
from robosuite.utils import transform_utils as T
from robosuite.utils.placement_samplers import UniformRandomSampler

H, W = 224, 384  # same frame size as the DINO pipeline
STEP = 0.05  # metres per unit of normalised OSC position action
ROT = 0.5  # radians per unit of normalised OSC rotation action
HOVER = 0.12
CAMERA_FOVY = 32  # zoom so objects span about two DINO patches
TRAIN_COLOURS = [(0.9, 0.1, 0.1), (0.1, 0.4, 0.9), (0.1, 0.7, 0.2), (0.95, 0.75, 0.1), (0.6, 0.2, 0.8), (0.95, 0.5, 0.1)]
TEST_COLOURS = [(0.1, 0.8, 0.8), (0.9, 0.3, 0.6), (0.4, 0.25, 0.1), (0.5, 0.5, 0.5)]  # held out, for the unseen-object evaluation
SHAPES = ("cube", "cuboid", "cylinder")


class ClutterScene(Lift):
    """Lift's table scene with a target cube, cuboid or cylinder plus random primitive distractors, all placed at random."""

    def __init__(self, shape, half, colour, distractors, **kwargs):
        self.shape, self.half, self.colour, self.distractor_specs = shape, half, colour, distractors
        super().__init__(**kwargs)

    def _load_model(self):
        """Build the scene: table, robot, the target object (named cube so Lift's observables work) and the distractors."""
        SingleArmEnv._load_model(self)
        self.robots[0].robot_model.set_base_xpos(self.robots[0].robot_model.base_xpos_offset["table"](self.table_full_size[0]))
        arena = TableArena(table_full_size=self.table_full_size, table_friction=self.table_friction, table_offset=self.table_offset)
        arena.set_origin([0, 0, 0])
        rgba = [*self.colour, 1]
        self.cube = CylinderObject("cube", size=self.half[[0, 2]], rgba=rgba) if self.shape == "cylinder" else BoxObject("cube", size=self.half, rgba=rgba)
        self.distractors = [make_primitive(f"d{i}", *spec) for i, spec in enumerate(self.distractor_specs)]
        self.placement_initializer = UniformRandomSampler(
            name="Sampler", mujoco_objects=[self.cube, *self.distractors], x_range=[-0.1, 0.1], y_range=[-0.2, 0.2], rotation=None,
            ensure_object_boundary_in_range=False, ensure_valid_placement=True, reference_pos=self.table_offset, z_offset=0.01,
        )
        self.model = ManipulationTask(mujoco_arena=arena, mujoco_robots=[r.robot_model for r in self.robots], mujoco_objects=[self.cube, *self.distractors])

    def _setup_references(self):
        """Also look up the distractors' body ids."""
        super()._setup_references()
        self.distractor_ids = [self.sim.model.body_name2id(d.root_body) for d in self.distractors]

    def distractor_pos(self):
        """Return the (n, 3) world positions of the distractors."""
        return self.sim.data.body_xpos[self.distractor_ids].copy()


def make_primitive(name, kind, size, colour):
    """Build a box, cylinder or ball distractor of the given half-size and colour."""
    rgba = [*colour, 1]
    if kind == "ball":
        return BallObject(name, size=[size[0]], rgba=rgba)
    return CylinderObject(name, size=[size[0], size[2]], rgba=rgba) if kind == "cylinder" else BoxObject(name, size=size, rgba=rgba)


def sample_object(shape, rng):
    """Return (half-extents, colour) for the target; a cuboid has one long and one short horizontal side."""
    c = TRAIN_COLOURS[rng.integers(len(TRAIN_COLOURS))]
    if shape == "cube":
        s = rng.uniform(0.02, 0.026)
        return np.array([s, s, s]), c
    if shape == "cuboid":
        return np.array([rng.uniform(0.016, 0.02), rng.uniform(0.03, 0.04), rng.uniform(0.02, 0.026)]), c
    return np.array([rng.uniform(0.02, 0.026), 0.0, rng.uniform(0.03, 0.04)]), c


def sample_distractors(rng):
    """Return 3 to 5 random (kind, half-extents, colour) specs."""
    specs = []
    for _ in range(rng.integers(3, 6)):
        kind = ("box", "cylinder", "ball")[rng.integers(3)]
        s = rng.uniform(0.015, 0.025)
        size = np.array([s, s * rng.uniform(0.8, 1.6), s * rng.uniform(0.8, 1.6)])
        specs.append((kind, size, TRAIN_COLOURS[rng.integers(len(TRAIN_COLOURS))]))
    return specs


def sample_target(start, others, rng):
    """Draw a random table position far from the start and from every other object."""
    while True:
        p = np.array([rng.uniform(-0.1, 0.1), rng.uniform(-0.2, 0.2)])
        if np.linalg.norm(p - start[:2]) > 0.12 and all(np.linalg.norm(p - o[:2]) > 0.07 for o in others):
            return p


def yaw_error(obs, half, shape, mode):
    """Signed yaw (rad) that turns the fingers to close across a face of the object, modulo the gripper's 180-degree symmetry."""
    if shape == "cylinder":
        return 0.0
    axes = T.quat2mat(obs["cube_quat"])[:2, :2].T  # object x and y axes in the world xy plane
    cur = T.quat2mat(obs["robot0_eef_quat"])[:2, 1]
    want = axes[int(np.argmin(half[:2]))] if shape == "cuboid" else axes[0]
    period = np.pi / 2 if shape == "cube" else np.pi  # a cube looks the same every 90 degrees, a cuboid every 180
    angle = np.arctan2(want[1], want[0]) - np.arctan2(cur[1], cur[0]) + (np.pi / 4 if mode == "edge" else 0)
    return (angle + period / 2) % period - period / 2


class Recorder:
    """Steps the environment and stores per-step observations, gripper state and the executed action."""

    def __init__(self, env, obs):
        self.env, self.obs = env, obs
        self.data = {k: [] for k in ("rgb", "eef_pos", "eef_quat", "gripper_qpos", "object_pos", "object_quat", "distractor_pos", "action")}
        self.snapshot()

    def snapshot(self):
        """Append the current observation."""
        o = self.obs
        for key, value in (("rgb", o["agentview_image"][::-1]), ("eef_pos", o["robot0_eef_pos"]), ("eef_quat", o["robot0_eef_quat"]),
                           ("gripper_qpos", o["robot0_gripper_qpos"]), ("object_pos", o["cube_pos"]), ("object_quat", o["cube_quat"]),
                           ("distractor_pos", self.env.distractor_pos())):
            self.data[key].append(np.array(value))

    def step(self, action):
        """Apply a 7-d action (dx, dy, dz, drot x, y, z, gripper) and record the result."""
        self.data["action"].append(np.array(action, dtype=np.float32))
        self.obs, *_ = self.env.step(action)
        self.snapshot()

    def move(self, goal, grip, mode=None, half=None, shape=None, hold=0, release_at=None, max_steps=90):
        """Drive the end effector to goal while turning to the grasp yaw (if mode is given); optionally open the gripper partway."""
        start = np.linalg.norm(goal - self.obs["robot0_eef_pos"]) + 1e-6
        for t in range(max_steps):
            err = goal - self.obs["robot0_eef_pos"]
            yaw = yaw_error(self.obs, half, shape, mode) if mode else 0.0
            if np.linalg.norm(err) < 0.008 and abs(yaw) < 0.05 and t >= hold:
                return
            if release_at is not None and 1 - np.linalg.norm(err) / start > release_at:
                grip = -1
            self.step(np.concatenate([np.clip(err / STEP * 2, -1, 1), [0, 0, np.clip(yaw / ROT * 2, -1, 1)], [grip]]))
        return

    def wait(self, grip, steps):
        """Hold still for a number of steps."""
        for _ in range(steps):
            self.step(np.concatenate([np.zeros(6), [grip]]))


def run_episode(env, obs, target, mode, half, shape, top_frac, miss_offset):
    """Execute hover, align, descend, grasp, lift, carry, lower, release and retreat; return the recorder and the object start position."""
    rec = Recorder(env, obs)
    obj = obs["cube_pos"].copy()
    grasp_z = obj[2] + (top_frac * half[2] if mode == "edge" else 0.005)
    above = lambda p, z: np.array([p[0], p[1], z])
    kw = dict(mode=mode, half=half, shape=shape)
    at = obj + np.append(miss_offset, 0.0) if mode == "miss" else obj  # a miss closes the gripper beside the object
    rec.move(above(at, obj[2] + HOVER), -1, **kw)
    rec.move(above(at, grasp_z), -1, **kw)
    rec.move(above(at, grasp_z), 1, hold=12)
    rec.move(above(at, obj[2] + HOVER), 1)
    rec.move(above(target, obj[2] + HOVER), 1, release_at=0.35 if mode == "release" else None)
    rec.move(above(target, grasp_z + 0.008), -1 if mode == "release" else 1)
    rec.wait(-1, 10)
    rec.move(above(target, obj[2] + HOVER), -1)
    return rec, obj


def label(rec, obj, target):
    """Return the observed outcome: placed upright at the target, dropped after being lifted, or never lifted."""
    final = rec.data["object_pos"][-1]
    upright = T.quat2mat(rec.data["object_quat"][-1])[2, 2] > np.cos(np.radians(20))
    lifted = max(p[2] for p in rec.data["object_pos"]) > obj[2] + 0.05
    placed = np.linalg.norm(final[:2] - target) < 0.03 and upright
    return ("placed" if placed else "dropped" if lifted else "not_lifted"), float(np.linalg.norm(final[:2] - target))


def hide_collision_geoms(env):
    """Make the robot's collision geoms invisible; robosuite draws them as solid green and blue paint over the visual meshes in this setup."""
    m = env.sim.model
    ids = [g for g in range(m.ngeom) if m.geom_group[g] == 0 and (m.body_id2name(m.geom_bodyid[g]) or "").startswith(("robot0_", "gripper0_", "mount0_", "base0_"))]
    m.geom_rgba[ids, 3] = 0


def make_env(shape, half, colour, distractors, gripper="default"):
    """Create one headless Panda scene (with the given robosuite gripper, the Panda's own by default) with the OSC end-effector controller and a zoomed agentview camera."""
    env = ClutterScene(shape, half, colour, distractors, robots="Panda", gripper_types=gripper, controller_configs=load_controller_config(default_controller="OSC_POSE"),
                       has_renderer=False, has_offscreen_renderer=True, use_camera_obs=True, camera_names="agentview", camera_heights=H,
                       camera_widths=W, control_freq=20, horizon=4000, ignore_done=True, hard_reset=False)
    env.sim.model.cam_fovy[env.sim.model.camera_name2id("agentview")] = CAMERA_FOVY
    hide_collision_geoms(env)
    return env


def body_mask(env, prefixes):
    """Return a boolean (H, W) mask, upright like the recorded RGB, of the pixels where the agentview camera sees a body whose name starts with one of `prefixes`."""
    seg = env.sim.render(width=W, height=H, camera_name="agentview", segmentation=True)[::-1]
    model = env.sim.model
    wanted = np.array([(model.body_id2name(model.geom_bodyid[g]) or "").startswith(tuple(prefixes)) for g in range(model.ngeom)])
    return (seg[..., 0] == 5) & wanted[seg[..., 1]]  # 5 is the geom object type


def save_episode(path, rec, meta):
    """Write one episode's arrays and attributes to an HDF5 file, atomically so an interrupted run never leaves a broken file."""
    tmp = path.with_suffix(".tmp")
    with h5py.File(tmp, "w") as f:
        for key, values in rec.data.items():
            f.create_dataset(key, data=np.stack(values), compression="gzip" if key == "rgb" else None)
        f.attrs.update(meta)
    tmp.rename(path)


def collect(i, mode, seed, out):
    """Generate episode i (its own random stream, so reruns reproduce it); edge and release demos retry until the object is lifted and then dropped."""
    rng = np.random.default_rng([seed, i])
    shape = SHAPES[i % len(SHAPES)]
    half, colour = sample_object(shape, rng)
    for attempt in range(1, 7 if mode in ("edge", "release") else 2):
        env = make_env(shape, half, colour, sample_distractors(rng))
        np.random.seed(seed * 100000 + i * 10 + attempt)
        obs = env.reset()
        target = sample_target(obs["cube_pos"], env.distractor_pos(), rng)
        angle = rng.uniform(0, 2 * np.pi)
        miss = (max(half[:2]) + 0.035) * np.array([np.cos(angle), np.sin(angle)])
        rec, obj = run_episode(env, obs, target, mode, half, shape, rng.uniform(0.7, 0.9), miss)
        outcome, dist = label(rec, obj, target)
        if mode not in ("edge", "release") or outcome == "dropped":
            break
        env.close()
    name = f"ep_{i:03d}"
    meta = dict(shape=shape, mode=mode, outcome=outcome, attempts=attempt, final_dist_to_target=dist, half=half, colour=colour, target=target, n_distractors=len(env.distractors))
    save_episode(out / "episodes" / f"{name}.h5", rec, meta)
    imageio.mimwrite(out / "videos" / f"{name}.mp4", rec.data["rgb"], fps=20)
    env.close()
    return {"episode": name, "shape": shape, "mode": mode, "outcome": outcome, "dist_cm": round(dist * 100, 1), "steps": len(rec.data["action"]), "attempts": attempt}


def main():
    """Collect the episodes into --out/episodes/*.h5 (skipping finished ones, so reruns resume), with videos/ and metadata.csv."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--release", type=int, default=3, help="demos that open the gripper mid-carry, retried until they drop")
    parser.add_argument("--edge", type=int, default=3, help="demos that grasp at a corner or the top edge, retried until they drop")
    parser.add_argument("--miss", type=int, default=3, help="demos whose gripper closes beside the object")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--worker", type=int, default=0)
    parser.add_argument("--workers", type=int, default=1, help="split the episodes across this many parallel processes; rerun with 1 to write metadata.csv")
    args = parser.parse_args()
    out = Path(args.out)
    for sub in ("episodes", "videos"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    special = ["release"] * args.release + ["edge"] * args.edge + ["miss"] * args.miss
    modes = special + ["normal"] * (args.episodes - len(special))
    modes = [modes[i] for i in np.random.default_rng(args.seed).permutation(len(modes))]
    for i, mode in enumerate(modes):
        if i % args.workers == args.worker and not (out / "episodes" / f"ep_{i:03d}.h5").exists():
            print(collect(i, mode, args.seed, out), flush=True)
    rows = []
    for path in sorted((out / "episodes").glob("ep_*.h5")):
        with h5py.File(path) as f:
            a = f.attrs
            rows.append({"episode": path.stem, "shape": a["shape"], "mode": a["mode"], "outcome": a["outcome"], "dist_cm": round(a["final_dist_to_target"] * 100, 1), "steps": len(f["action"]), "attempts": a["attempts"]})
    with open(out / "metadata.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
