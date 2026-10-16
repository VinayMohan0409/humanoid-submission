"""Task evaluation: each condition (world model, IDM, gripper) runs the planner on fixed-seed open-gripper reach tasks and carry tasks, success being the end effector within 5 cm of the goal."""
import argparse
import json

import imageio
import numpy as np
import torch

from src.config import ARTIFACT_DIR, CHECKPOINT_DIR
from src.planning.reach import SUCCESS, dino_features, load_models, make_goal, play_prior, run
from src.plotting import overlay_silhouette

CONDITIONS = {
    "human_only_panda": {"world": "pretrain_human", "idm": "idm_panda", "gripper": "default"},
    "cotrain_panda": {"world": "cotrain", "idm": "idm_panda", "gripper": "default"},
    "cotrain_robotiq": {"world": "cotrain", "idm": "idm_robotiq", "gripper": "Robotiq85Gripper"},
}
DIRECTIONS = 5  # goal direction bins across the forward +-90 degree arc, the only directions the workspace allows from the start pose
TASKS = {  # carry: start holding a cube, cuboid or cylinder; the IDM has no gripper output, so the gripper stays at hold_grip
    "reach": {"carry": False, "goal_grip": -1, "hold_grip": -1.0, "seeds": [0, 1, 2, 3, 4]},
    "carry": {"carry": True, "goal_grip": 1, "hold_grip": 1.0, "seeds": list(range(10))},
}
OUT = ARTIFACT_DIR / "reach" / "task_eval"


def run_task(condition, task, seed, goal, features, model, idm, prior, steps):
    """Run the planner on one task's goal, save its video with the green goal silhouette, and return the result row."""
    torch.manual_seed(seed)
    rec, distances = run(goal["reset"], goal["goal"], goal["pos"], features, model, idm, prior, steps // 4, hold_grip=TASKS[task]["hold_grip"])
    folder = OUT / condition / task / f"seed_{seed}"
    folder.mkdir(parents=True, exist_ok=True)
    imageio.mimwrite(folder / "planner.mp4", overlay_silhouette(np.array(rec.data["rgb"]), goal["mask"]), fps=20)
    imageio.imwrite(folder / "goal.png", overlay_silhouette(goal["rgb"][None], goal["mask"])[0])
    return {"condition": condition, "task": task, "seed": seed, "start_m": distances[0], "end_m": distances[-1], "closest_m": min(distances),
            "success": bool(min(distances) < SUCCESS), "replans": len(distances) - 1, "held": goal["held"]}


def evaluate_condition(condition, spec, features, steps):
    """Return the result rows of every task and seed for one condition."""
    model, idm = load_models(CHECKPOINT_DIR / spec["world"] / "last.pt", CHECKPOINT_DIR / spec["idm"] / "best.pt")
    prior = play_prior(model)
    rows = []
    (OUT / condition).mkdir(parents=True, exist_ok=True)
    for task, t in TASKS.items():
        for seed in t["seeds"]:
            try:
                goal = make_goal(seed, features, spec["gripper"], t["goal_grip"], t["carry"], (seed % DIRECTIONS, DIRECTIONS))
            except RuntimeError as error:
                print(f"{condition:18s} {task:6s} seed {seed:2d} skipped: {error}", flush=True)
                continue
            rows.append(run_task(condition, task, seed, goal, features, model, idm, prior, steps))
            r = rows[-1]
            print(f"{condition:18s} {task:6s} seed {seed:2d} start {r['start_m'] * 100:5.1f} end {r['end_m'] * 100:5.1f} closest {r['closest_m'] * 100:5.1f} cm  {'SUCCESS' if r['success'] else 'fail'}", flush=True)
            goal["env"].close()
            json.dump(rows, open(OUT / condition / "results.json", "w"), indent=1)
    return rows


def main():
    """Evaluate one condition and write its results.json under artifacts/reach/task_eval/<condition>."""
    parser = argparse.ArgumentParser()
    parser.add_argument("condition", choices=list(CONDITIONS))
    parser.add_argument("--steps", type=int, default=300, help="control steps allowed, 0.05 s each")
    parser.add_argument("--carry-seeds", type=int, nargs="+", help="override the carry task seeds")
    args = parser.parse_args()
    if args.carry_seeds:
        TASKS["carry"]["seeds"] = args.carry_seeds
    rows = evaluate_condition(args.condition, CONDITIONS[args.condition], dino_features(), args.steps)
    json.dump(rows, open(OUT / args.condition / "results.json", "w"), indent=1)


if __name__ == "__main__":
    main()
