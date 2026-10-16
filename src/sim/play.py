"""Generate task-agnostic robot play data: the Panda moves through random waypoints and toggles its gripper among clutter, saving actions and gripper state per episode."""
import argparse
from pathlib import Path

import imageio
import numpy as np

from src.sim import scenes as S

STEPS = 100
X, Y, Z = (-0.12, 0.12), (-0.22, 0.22), (0.83, 1.08)  # waypoint ranges above the table, metres
WAYPOINT_STEPS = 25


def play(env, obs, rng):
    """Drive the arm through random waypoints at random speeds with small noise, random yaw and random gripper toggles; return the recorder."""
    rec = S.Recorder(env, obs)
    grip, goal, gain, yaw = -1.0, None, 1.0, 0.0
    for t in range(STEPS):
        pos = rec.obs["robot0_eef_pos"]
        if goal is None or t % WAYPOINT_STEPS == 0 or np.linalg.norm(goal - pos) < 0.01:
            goal = np.array([rng.uniform(*X), rng.uniform(*Y), rng.uniform(*Z)])
            gain, yaw = rng.uniform(0.5, 3.0), rng.uniform(-0.4, 0.4) * (rng.random() < 0.4)
            grip = -grip if rng.random() < 0.3 else grip
        move = np.clip((goal - pos) / S.STEP * gain, -1, 1) + rng.normal(0, 0.05, 3)
        rec.step(np.concatenate([np.clip(move, -1, 1), [0, 0, yaw], [grip]]))
    return rec


def collect(i, seed, out, gripper):
    """Generate episode i in a random scene (its own random stream, so reruns reproduce it) and save it."""
    rng = np.random.default_rng([seed, 10_000 + i])
    shape = S.SHAPES[i % len(S.SHAPES)]
    half, colour = S.sample_object(shape, rng)
    env = S.make_env(shape, half, colour, S.sample_distractors(rng), gripper)
    np.random.seed(seed * 100000 + i)
    rec = play(env, env.reset(), rng)
    meta = dict(shape=shape, gripper=gripper, mode="play", outcome="none", attempts=1, final_dist_to_target=0.0, half=half, colour=colour, target=np.zeros(2), n_distractors=len(env.distractors))
    S.save_episode(out / "episodes" / f"ep_{i:04d}.h5", rec, meta)
    if i < 8:
        imageio.mimwrite(out / "videos" / f"ep_{i:04d}.mp4", rec.data["rgb"], fps=20)
    env.close()


def main():
    """Collect this worker's share of the episodes into --out/episodes, skipping finished ones so reruns resume."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--worker", type=int, default=0)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--gripper", default="default", help="robosuite gripper name, e.g. Robotiq85Gripper; default is the Panda's own")
    args = parser.parse_args()
    out = Path(args.out)
    for sub in ("episodes", "videos"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    for i in range(args.worker, args.episodes, args.workers):
        if not (out / "episodes" / f"ep_{i:04d}.h5").exists():
            collect(i, args.seed, out, args.gripper)
            print(f"ep_{i:04d} done", flush=True)


if __name__ == "__main__":
    main()
