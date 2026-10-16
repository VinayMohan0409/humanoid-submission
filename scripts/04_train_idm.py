"""Train the inverse dynamics model on action-labelled play episodes, keeping the checkpoint with the best held-out play error."""
import argparse
import json
import time
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from src.config import CHECKPOINT_DIR, ENC_LAYERS, GRAD_CLIP, PLAY_DIR, SEED, SIM_STRIDE, WEIGHT_DECAY, get_device
from src.datasets.sim import DEMOS, PLAY, SimChunkDataset
from src.models.idm import InverseDynamics, action_columns, pairs

BATCH_SIZE = 32  # windows per step, each giving CHUNK frame pairs


def forward(model, batch, device):
    """Return the predicted and true actions of every frame pair in the batch."""
    before, after, truth = pairs(batch, model.outputs)
    with torch.autocast(device, torch.bfloat16, enabled=device == "cuda"):
        return model(before, after).float(), truth


@torch.no_grad()
def evaluate(model, data, mean_action, device):
    """Return the position MSE, the mean-action baseline's MSE and, if the IDM predicts the gripper, how often its sign matches the true command next to the majority-sign guess."""
    model.eval()
    sq = base = correct = closed = 0
    for batch in data.batches(128, shuffle=False):
        predicted, truth = forward(model, batch, device)
        sq, base = sq + (predicted - truth).pow(2).sum((0, 1)), base + (truth - mean_action).pow(2).sum((0, 1))
        if model.outputs == 4:
            correct, closed = correct + ((predicted[..., 3] > 0) == (truth[..., 3] > 0)).sum().item(), closed + (truth[..., 3] > 0).sum().item()
    n = len(data) * 5 * SIM_STRIDE
    result = {"mse": (sq[:3].sum() / (n * 3)).item(), "baseline_mse": (base[:3].sum() / (n * 3)).item()}
    return {**result, "gripper_accuracy": correct / n, "gripper_majority": max(closed, n - closed) / n} if model.outputs == 4 else result


def main():
    """Train on the first --max-episodes play episodes, evaluating on held-out play and demo episodes each epoch; keep the best play-val checkpoint."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="idm")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--max-episodes", type=int, default=50)
    parser.add_argument("--data", default=str(PLAY_DIR), help="play-data folder with episodes/ and features/; every 10th episode is held out")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--d-model", type=int, default=256, help="transformer width")
    parser.add_argument("--dropout", type=float, default=0.0, help="dropout in every transformer layer and the head")
    parser.add_argument("--layers", type=int, default=ENC_LAYERS, help="number of transformer layers")
    parser.add_argument("--head-hidden", type=int, default=0, help="width of a hidden layer in the output head; 0 is a single linear layer")
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY)
    parser.add_argument("--gripper", action="store_true", help="also predict the gripper command (open or closed)")
    args = parser.parse_args()
    device = get_device()
    torch.manual_seed(SEED)
    run_dir = CHECKPOINT_DIR / args.name
    run_dir.mkdir(parents=True, exist_ok=True)
    model = InverseDynamics(args.d_model, args.dropout, args.layers, args.head_hidden, args.gripper).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    play = (Path(args.data), PLAY[1])
    train = SimChunkDataset("train", device, (play,), args.max_episodes)
    val = {"play": SimChunkDataset("val", device, (play,)), "demos": SimChunkDataset("val", device, (DEMOS,))}
    mean_action = train.actions[:, action_columns(model.outputs)].mean(0)
    print({"train windows": len(train), **{f"{k} val windows": len(v) for k, v in val.items()}}, flush=True)

    best = float("inf")
    for epoch in range(args.epochs):
        start, total, steps = time.time(), 0.0, 0
        model.train()
        for batch in train.batches(BATCH_SIZE, shuffle=True):
            predicted, truth = forward(model, batch, device)
            loss = F.mse_loss(predicted, truth)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            total, steps = total + loss.item(), steps + 1
        results = {name: evaluate(model, data, mean_action, device) for name, data in val.items()}
        print(f"epoch {epoch}: {time.time() - start:.0f}s train mse {total / steps:.4f} | " + " | ".join(
            f"{n} val position mse {m['mse']:.4f} (baseline {m['baseline_mse']:.4f})" + (f", gripper right {m['gripper_accuracy']:.3f} (majority guess {m['gripper_majority']:.3f})" if "gripper_accuracy" in m else "") for n, m in results.items()), flush=True)
        with open(run_dir / "metrics.jsonl", "a") as f:
            f.write(json.dumps({"epoch": epoch, "train_mse": total / steps, **{f"{n}_val": m for n, m in results.items()}}) + "\n")
        if results["play"]["mse"] < best:
            best = results["play"]["mse"]
            torch.save({"model": model.state_dict(), "epoch": epoch, "args": vars(args)}, run_dir / "best.pt")


if __name__ == "__main__":
    main()
