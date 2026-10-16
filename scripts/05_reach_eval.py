"""Reach evaluation: from the same start, each planner variant drives the end effector toward a goal frame rendered by scripted motion; reports distance, smoothness and gripper behaviour."""
import argparse
import json
from pathlib import Path

import imageio
import numpy as np
import torch

from src.config import ARTIFACT_DIR, CHECKPOINT_DIR
from src.planning.reach import SUCCESS, dino_features, finger_closure, load_models, make_goal, play_prior, run
from src.plotting import MUTED, capitalise, finish, new_figure, overlay_silhouette, style_axes

DEFAULTS = {"weighted": False, "execute": 4, "metric": "l2", "idm": "idm_panda", "gripper": "default", "goal_grip": -1, "hold": False}  # a variant overrides any of these
ROBOTIQ = {"gripper": "Robotiq85Gripper", "idm": "idm_robotiq"}  # a Robotiq 85 gripper on the Panda arm, with the IDM trained on its play data
VARIANT_SETS = {
    "base": {"planner": {}},
    "smooth": {"best-frame cost, execute 4": {}, "weighted cost, execute 4": {"weighted": True}, "weighted cost, execute 2": {"weighted": True, "execute": 2}, "weighted cost, execute 1": {"weighted": True, "execute": 1}},
    "hold": {"no hold": {}, "hold when staying is better": {"hold": True}},
    "metric": {"squared error (L2)": {}, "cosine distance": {"metric": "cosine"}, "L2 + cosine": {"metric": "both"}},
    "robotiq": {"Robotiq, open goal": ROBOTIQ, "Robotiq, closed goal": {**ROBOTIQ, "goal_grip": 1}},
}
STEPS = 100  # default control steps allowed (5 s)


def idm_path(name):
    """Return the best.pt path of a trained IDM checkpoint folder."""
    return CHECKPOINT_DIR / name / "best.pt"


def smoothness(eef):
    """Return the path length over the net displacement (1 = a straight line) and the mean change of velocity per control step in mm per step^2 (lower is smoother) of an end-effector track."""
    v = np.diff(eef, axis=0)
    return float(np.linalg.norm(v, axis=1).sum() / (np.linalg.norm(eef[-1] - eef[0]) + 1e-9)), float(np.linalg.norm(np.diff(v, axis=0), axis=1).mean() * 1000)


def gripper_stats(rec, g):
    """Return the share of the last 20 gripper commands that were closed, and how far the fingers got toward the goal's finger positions (1 = as closed as the goal), or None if the goal's gripper is open."""
    command = np.array(rec.data["action"])[:, 6]
    open_goal = np.linalg.norm(g["qpos_open"] - g["qpos_goal"]) < 1e-3
    return float((command[-20:] > 0).mean()), None if open_goal else finger_closure(np.array(rec.data["gripper_qpos"])[-1], g["qpos_open"], g["qpos_goal"])


def save_plot(path, results):
    """Plot each variant's distance to the goal, end-effector speed and gripper command against the control step."""
    fig, axes = new_figure(1, 3, (18, 4.8))
    ax = axes[0]
    for name, r in results.items():
        ax[0].plot(np.array(r["distance_m"]) * 100, label=capitalise(name))
        ax[1].plot(np.array(r["speed_m"]) * 1000, label=capitalise(name), alpha=0.8)
        ax[2].plot(r["command"], label=capitalise(name), alpha=0.8)
    ax[0].axhline(SUCCESS * 100, color=MUTED, ls=":", label=f"Success ({SUCCESS * 100:.0f} Centimetres)")
    style_axes(ax[0], xlabel="Control Step", ylabel="Distance To Goal (Centimetres)")
    style_axes(ax[1], xlabel="Control Step", ylabel="End-Effector Speed (Millimetres Per Step)")
    style_axes(ax[2], xlabel="Control Step", ylabel="Gripper Command (-1 Open, 1 Closed)")
    ax[0].set_ylim(0, None)
    ax[1].set_ylim(0, None)
    ax[2].set_ylim(-1.2, 1.2)
    ax[0].legend(fontsize=8, frameon=False)
    finish(fig, path, caption="One control step is 0.05 seconds.", dpi=120)


def main():
    """Run every variant of the chosen set on one scene and report distance, smoothness and gripper behaviour, saving videos, goal frames, a plot and results.json."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--set", default="gripper", choices=list(VARIANT_SETS), help="which group of variants to compare")
    parser.add_argument("--steps", type=int, default=STEPS, help="control steps allowed, 0.05 s each")
    parser.add_argument("--cotrain", default=str(CHECKPOINT_DIR / "cotrain" / "last.pt"))
    parser.add_argument("--out", default=None, help="default: artifacts/reach/<set>")
    args = parser.parse_args()
    out = Path(args.out or ARTIFACT_DIR / "reach" / args.set) / f"scene_{args.seed}"
    out.mkdir(parents=True, exist_ok=True)

    features, models = dino_features(), {}
    prior = play_prior(load_models(args.cotrain, idm_path(DEFAULTS["idm"]))[0])
    print(f"scene {args.seed}, set '{args.set}'", flush=True)
    print(f"{'variant':26s} {'steps':>5s} {'end cm':>7s} {'closest':>8s} {'path/straight':>14s} {'accel':>6s} | {'closed in last 20 steps':>24s} {'fingers vs goal':>16s} {'held':>5s}")
    results = {}
    for name, override in VARIANT_SETS[args.set].items():
        o = {**DEFAULTS, **override}
        if o["idm"] not in models:
            models[o["idm"]] = load_models(args.cotrain, idm_path(o["idm"]))
        model, idm = models[o["idm"]]
        g = make_goal(args.seed, features, o["gripper"], o["goal_grip"])
        torch.manual_seed(args.seed)
        rec, _ = run(g["reset"], g["goal"], g["pos"], features, model, idm, prior, args.steps // o["execute"], o["execute"], o["weighted"], o["metric"], (g["qpos_open"], g["qpos_goal"]), o["hold"])
        eef = np.array(rec.data["eef_pos"])
        distance = np.linalg.norm(eef - g["pos"], axis=1)
        ratio, accel = smoothness(eef)
        closed, closure = gripper_stats(rec, g)
        held = int(np.all(np.array(rec.data["action"])[:, :3] == 0, axis=1).sum())  # control steps with zero position command
        results[name] = {"distance_m": distance.tolist(), "speed_m": np.linalg.norm(np.diff(eef, axis=0), axis=1).tolist(), "command": np.array(rec.data["action"])[:, 6].tolist(),
                         "path_over_straight": ratio, "accel_mm_per_step2": accel, "closed_last20": closed, "finger_closure": closure, "held_steps": held}
        print(f"{name:26s} {len(eef) - 1:5d} {distance[-1] * 100:7.1f} {distance.min() * 100:8.1f} {ratio:14.2f} {accel:6.2f} | {closed:24.0%} {'n/a (goal open)' if closure is None else f'{closure:.0%}':>16s} {held:5d}", flush=True)
        slug = name.replace(", ", "_").replace(" ", "_")
        imageio.mimwrite(out / f"{slug}.mp4", overlay_silhouette(np.array(rec.data["rgb"]), g["mask"]), fps=20)
        imageio.imwrite(out / f"goal_{slug}.png", overlay_silhouette(g["rgb"][None], g["mask"])[0])
        g["env"].close()
    json.dump({"seed": args.seed, "variants": results}, open(out / "results.json", "w"))
    save_plot(out / "compare.png", results)


if __name__ == "__main__":
    main()
