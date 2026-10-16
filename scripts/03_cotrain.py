"""Co-train one latent action model from scratch on human clips and simulated Panda demonstrations, logging train and held-out curves for each."""
import argparse
import json
import time

import torch
from torch import nn
from torch.nn import functional as F

from src.config import CHECKPOINT_DIR, GRAD_CLIP, LR, SEED, WEIGHT_DECAY, get_device
from src.datasets.human import ChunkDataset
from src.datasets.sim import SimFrames
from src.models.latent_action import LatentActionEncoder
from src.models.world_model import WorldModel
from src.training.evaluation import evaluate, run_model

HALF_BATCH = 16  # windows per domain per step, so a co-trained step sees 32


def cycle(data, batch_size):
    """Yield shuffled batches forever, reshuffling after each pass."""
    while True:
        yield from data.batches(batch_size, shuffle=True)


def window_loss(encoder, world_model, batch):
    """Return the feature-prediction MSE of one batch of frame windows."""
    pred, _, target, _, _ = run_model(encoder, world_model, batch, sample=True)
    return F.mse_loss(pred, target)


def main():
    """Train for --epochs, evaluating every dataset's train and validation windows after each epoch and appending them to metrics.jsonl."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="cotrain_30demos")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--robot-only", action="store_true", help="drop the human clips from training (control); the human clips are still evaluated")
    parser.add_argument("--resume", action="store_true", help="continue from last.pt in the run folder if it exists")
    args = parser.parse_args()

    device = get_device()
    torch.manual_seed(SEED)
    run_dir = CHECKPOINT_DIR / args.name
    run_dir.mkdir(parents=True, exist_ok=True)
    encoder, world_model = LatentActionEncoder().to(device), WorldModel().to(device)
    model = nn.ModuleDict({"encoder": encoder, "world_model": world_model})
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    data = {"human": {s: ChunkDataset(s, device) for s in ("train", "val")}, "sim": {s: SimFrames(s, device) for s in ("train", "val")}}
    print({d: {s: len(x) for s, x in splits.items()} for d, splits in data.items()}, "windows", flush=True)
    sim_batches = cycle(data["sim"]["train"], HALF_BATCH * (2 if args.robot_only else 1))
    start_epoch = 0
    if args.resume and (run_dir / "last.pt").exists():
        saved = torch.load(run_dir / "last.pt", map_location=device)
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        start_epoch = saved["epoch"] + 1
        kept = [line for line in open(run_dir / "metrics.jsonl") if json.loads(line)["epoch"] < start_epoch]
        open(run_dir / "metrics.jsonl", "w").writelines(kept)
        print(f"resumed at epoch {start_epoch} of {args.epochs}", flush=True)

    for epoch in range(start_epoch, args.epochs):
        start = time.time()
        model.train()
        sums, steps = {"human": 0.0, "sim": 0.0}, 0
        for human_batch in data["human"]["train"].batches(HALF_BATCH, shuffle=True, drop_last=True):
            batches = {"sim": next(sim_batches).to(device)} if args.robot_only else {"human": human_batch, "sim": next(sim_batches).to(device)}
            losses = {name: window_loss(encoder, world_model, batch) for name, batch in batches.items()}
            loss = sum(losses.values()) / len(losses)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            for name, value in losses.items():
                sums[name] += value.item()
            steps += 1
        record = {"epoch": epoch, **{f"train_loss_{name}": total / steps for name, total in sums.items()}}
        for domain, splits in data.items():
            for split, dataset in splits.items():
                record[f"{domain}_{split}"] = evaluate(encoder, world_model, dataset, torch.Generator().manual_seed(SEED), device)
        ratio = lambda key: sum(record[key]["model"]) / 5
        print(f"epoch {epoch}: {time.time() - start:.0f}s | " + " | ".join(
            f"{d} train {ratio(f'{d}_train'):.3f} val {ratio(f'{d}_val'):.3f} shuffled {sum(record[f'{d}_val']['shuffled']) / 5:.2f}" for d in data) + "  (mse / copy-F0 mse)", flush=True)
        with open(run_dir / "metrics.jsonl", "a") as f:
            f.write(json.dumps(record) + "\n")
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch}, run_dir / "last.tmp")
        (run_dir / "last.tmp").replace(run_dir / "last.pt")


if __name__ == "__main__":
    main()
