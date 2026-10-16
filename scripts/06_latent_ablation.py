"""Latent ablation of the co-trained world model: does the world model use the latent action? Compares each window's own latent, another window's latent and one constant latent."""
import argparse
import csv
from pathlib import Path

import torch

from src.config import CHECKPOINT_DIR, FIGURE_DIR, LATENT_DIM, N_QUERIES, SEED, get_device
from src.datasets.human import ChunkDataset
from src.datasets.sim import SimFrames
from src.plotting import AQUA, BLUE, INK, ORANGE, finish, new_figure, style_axes
from src.training.checkpoints import load_model
from src.training.evaluation import latent_ablation, mean_latent

CONDITIONS = {"own": ("Own Latent", BLUE), "shuffled": ("Another Window's Latent", ORANGE), "constant": ("Same Latent For Every Window", AQUA)}
DOMAINS = {"human": ("Human Clips", ChunkDataset), "panda": ("Panda Demos", SimFrames)}


def evaluate_domain(model, dataset_class, device):
    """Return the per-step ratios of each condition on the domain's train and val windows, the constant being the mean latent of its train windows."""
    train, val = dataset_class("train", device), dataset_class("val", device)
    constant = mean_latent(model["encoder"], train)
    return {split: latent_ablation(model["encoder"], model["world_model"], data, constant, torch.Generator().manual_seed(SEED)) for split, data in (("train", train), ("val", val))}


def save_csv(path, results):
    """Write one row per domain, split and condition with the mean over the 5 predicted steps and each step's ratio."""
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["domain", "split", "condition", "mean_ratio", *[f"step_{k}" for k in range(1, 6)]])
        for domain, splits in results.items():
            for split, conditions in splits.items():
                for name, steps in conditions.items():
                    writer.writerow([domain, split, name, round(sum(steps) / len(steps), 3), *[round(x, 3) for x in steps]])


def save_plot(path, results):
    """Draw grouped bars of the held-out relative error per domain and condition."""
    fig, axes = new_figure(size=(7, 4.6))
    ax, width = axes[0, 0], 0.26
    for i, (name, (label, colour)) in enumerate(CONDITIONS.items()):
        values = [sum(results[d]["val"][name]) / 5 for d in DOMAINS]
        bars = ax.bar([x + (i - 1) * (width + 0.03) for x in range(len(DOMAINS))], values, width, color=colour, label=label)
        ax.bar_label(bars, fmt="%.2f", padding=2, fontsize=9, color=INK)
    ax.set_xticks(range(len(DOMAINS)), [label for label, _ in DOMAINS.values()])
    style_axes(ax, title=f"{N_QUERIES} Tokens × {LATENT_DIM} Dimensions", xlabel="Validation Set", ylabel="Relative Error")
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    finish(fig, path, caption="Relative error is compared with copy frame, which means the same frame being passed in.")


def main():
    """Evaluate every domain and write latent_ablation.csv and latent_ablation.png."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=str(CHECKPOINT_DIR / "cotrain" / "last.pt"))
    args = parser.parse_args()
    device = get_device()
    model = load_model(args.checkpoint, device)
    results = {domain: evaluate_domain(model, dataset_class, device) for domain, (_, dataset_class) in DOMAINS.items()}
    name = Path(args.checkpoint).parent.name
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    save_csv(FIGURE_DIR / f"latent_ablation_{name}.csv", results)
    save_plot(FIGURE_DIR / f"latent_ablation_{name}.png", results)
    print(FIGURE_DIR / f"latent_ablation_{name}.png")


if __name__ == "__main__":
    main()
