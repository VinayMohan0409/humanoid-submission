"""Pretrain the latent action encoder and world model on the human videos' cached DINO features."""
import argparse
import json
import time

import torch
from torch import nn
from torch.nn import functional as F
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
from torch.utils.tensorboard import SummaryWriter

from src.config import BETA, CHECKPOINT_DIR, EMA_DECAY, EPOCHS, GRAD_CLIP, KL_WARMUP, LR, SAMPLE_LATENT, SCALAR_EVERY, SEED, TRAIN_BATCH_SIZE, WEIGHT_DECAY, get_device
from src.datasets.human import ChunkDataset
from src.models.latent_action import LatentActionEncoder
from src.models.world_model import WorldModel
from src.training.checkpoints import load_checkpoint, save_checkpoint
from src.training.evaluation import evaluate, latent_stats, run_model


def main():
    """Train the encoder and world model on the train split, evaluating on the val split after every epoch."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="run")
    parser.add_argument("--epochs", type=int, default=EPOCHS, help="total number of epochs to reach, including resumed ones")
    parser.add_argument("--all-data", action="store_true", help="train on every clip; the val curves then cover clips the model has seen")
    parser.add_argument("--resume", action="store_true", help="continue from last.pt in the run folder")
    args = parser.parse_args()

    device = get_device()
    torch.manual_seed(SEED)
    run_dir = CHECKPOINT_DIR / args.name
    run_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(run_dir / "tb", flush_secs=5)

    encoder, world_model = LatentActionEncoder().to(device), WorldModel().to(device)
    model = nn.ModuleDict({"encoder": encoder, "world_model": world_model})
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    ema = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(EMA_DECAY)) if EMA_DECAY > 0 else None
    val_generator = torch.Generator().manual_seed(SEED)
    train_data, val_data = ChunkDataset("all" if args.all_data else "train", device), ChunkDataset("val", device)

    start_epoch, step, best = 0, 0, float("inf")
    warmup_steps = KL_WARMUP * (len(train_data) // TRAIN_BATCH_SIZE)
    if args.resume:
        start_epoch, step, best = load_checkpoint(run_dir / "last.pt", model, optimizer, val_generator, device)
        print(f"resumed at epoch {start_epoch}, training to epoch {args.epochs}")

    for epoch in range(start_epoch, args.epochs):
        start = time.time()
        model.train()
        total, norms, clipped = (torch.zeros((), device=device) for _ in range(3))
        for batch in train_data.batches(TRAIN_BATCH_SIZE, shuffle=True, drop_last=True):
            pred, _, target, _, kl = run_model(encoder, world_model, batch.to(device), sample=True)
            mse = F.mse_loss(pred, target)
            loss = mse + BETA * min(1.0, step / warmup_steps) * kl if warmup_steps else mse + BETA * kl
            optimizer.zero_grad()
            loss.backward()
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            if ema:
                ema.update_parameters(model)
            total += loss.detach()
            norms += grad_norm
            clipped += grad_norm > GRAD_CLIP
            if step % SCALAR_EVERY == 0:
                writer.add_scalar("train/loss", loss.item(), step)
                writer.add_scalar("train/mse", mse.item(), step)
                writer.add_scalar("train/kl", kl.item(), step)
            step += 1
        judged = ema.module if ema else model
        metrics = evaluate(judged["encoder"], judged["world_model"], val_data, val_generator, device)
        train_metrics = evaluate(judged["encoder"], judged["world_model"], train_data, torch.Generator().manual_seed(SEED), device)
        stats = latent_stats(judged["encoder"], val_data) if SAMPLE_LATENT else {}
        n = len(train_data) // TRAIN_BATCH_SIZE
        mean = lambda values: sum(values) / len(values)
        summary = {name: mean(metrics[name]) for name in ("model", "shuffled")}
        train_summary = mean(train_metrics["model"])
        train = {"train_loss": total.item() / n, "grad_norm": norms.item() / n, "clipped": clipped.item() / n}
        print(
            f"epoch {epoch}: {time.time() - start:.0f}s loss {train['train_loss']:.4f} | val mse {mean(metrics['mse']):.4f} "
            f"(copy-F0 {mean(metrics['mse_copy']):.4f}) train mse {mean(train_metrics['mse']):.4f} | mse / copy-mse: "
            f"train {train_summary:.3f} val {summary['model']:.3f} val-shuffled {summary['shuffled']:.3f} kl {metrics['kl']:.0f} "
            f"| val per step {[round(x, 2) for x in metrics['model']]} "
            f"| grad norm {train['grad_norm']:.3f} clipped {train['clipped']:.0%}"
        )
        if stats:
            z_mean, z_std = stats["z_mean"], stats["z_std"]
            print(f"  z over val: mean {min(z_mean):.2f}..{max(z_mean):.2f}, std {min(z_std):.2f}..{max(z_std):.2f}, active dims {stats['active_dims']}/{len(z_std)}, "
                  f"KL per pair {sum(stats['kl_per_dim']):.1f}, per dim {mean(stats['kl_per_dim']):.2f}")
        with open(run_dir / "metrics.jsonl", "a") as f:
            f.write(json.dumps({"epoch": epoch, **train, **metrics, **stats, "train_model": train_metrics["model"]}) + "\n")
        for name, value in train.items():
            writer.add_scalar(f"epoch/{name}", value, step)
        for name, value in {**summary, "kl": metrics["kl"]}.items():
            writer.add_scalar(f"val/{name}", value, step)
        writer.add_scalar("val/mse", mean(metrics["mse"]), step)
        writer.add_scalar("train_eval/mse", mean(train_metrics["mse"]), step)
        writer.add_scalar("train_eval/model", train_summary, step)
        for name in ("model", "shuffled"):
            for k, value in enumerate(metrics[name], 1):
                writer.add_scalar(f"val_step{k}/{name}", value, step)
        for k, value in enumerate(train_metrics["model"], 1):
            writer.add_scalar(f"train_step{k}/model", value, step)
        writer.flush()
        improved = summary["model"] < best
        best = min(best, summary["model"])
        save_checkpoint(run_dir / "last.pt", model, optimizer, epoch, step, best, val_generator, device, ema)
        if improved:
            save_checkpoint(run_dir / "best.pt", model, optimizer, epoch, step, best, val_generator, device, ema)
    writer.close()


if __name__ == "__main__":
    main()
