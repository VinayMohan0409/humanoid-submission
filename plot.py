"""Plot a run's training curves from its metrics.jsonl."""
import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import ROOT


def main():
    """Save curves.png in the run folder: train loss, validation error against the baselines, and KL, per epoch."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="run")
    run_dir = ROOT / "artifacts" / "runs" / parser.parse_args().name
    rows = [json.loads(line) for line in open(run_dir / "metrics.jsonl")]
    epochs = [row["epoch"] for row in rows]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(epochs, [row["train_loss"] for row in rows])
    axes[0].set(title="train loss (MSE + beta * KL)", xlabel="epoch", ylabel="loss")
    for key in ("model", "copy", "shuffled"):
        axes[1].plot(epochs, [row[key] for row in rows], label=key)
    axes[1].set(title="validation relative error", xlabel="epoch", ylabel="median relative error")
    axes[1].legend()
    axes[2].plot(epochs, [row["kl"] for row in rows])
    axes[2].set(title="validation KL", xlabel="epoch", ylabel="nats")
    fig.tight_layout()
    fig.savefig(run_dir / "curves.png", dpi=120)


if __name__ == "__main__":
    main()
