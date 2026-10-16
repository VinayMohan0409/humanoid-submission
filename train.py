"""Training entry point: pretrain on human video, finetune on LIBERO."""
import argparse
import json
import os
import time

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import config
from config import BETA, EPOCHS, GRAD_CLIP, LOG_EVERY, LR, ROOT, SCALAR_EVERY, SEED, TRAIN_BATCH_SIZE, WEIGHT_DECAY
from data.dataset import PairDataset
from diagnostics import Diagnostics
from models.latent_action import LatentActionEncoder, kl_divergence, sample_latent
from models.world_model import WorldModel


def relative_error(pred, target):
    """Return each pair's error as length(pred - target) / length(target), comparable to the copy-last-frame baseline."""
    return (pred - target).flatten(1).norm(dim=1) / target.flatten(1).norm(dim=1)


def run_model(encoder, world_model, batch, sample):
    """Encode (F_t, F_{t+h}) into a latent action z, then predict F_{t+h} from F_t and z; returns prediction, F_t, target, z and KL."""
    frame, target = batch[:, 0], batch[:, 1]
    mean, logvar = encoder(frame, target)
    z = sample_latent(mean, logvar) if sample else mean
    return world_model(frame, z), frame, target, z, kl_divergence(mean, logvar)


@torch.no_grad()
def evaluate(encoder, world_model, loader, device):
    """Return median relative error of the model, the copy-last-frame baseline and a shuffled latent, plus mean KL."""
    encoder.eval()
    world_model.eval()
    errors = {"model": [], "copy": [], "shuffled": []}
    kls = []
    for batch in loader:
        pred, frame, target, z, kl = run_model(encoder, world_model, batch.to(device), sample=False)
        errors["model"].append(relative_error(pred, target))
        errors["copy"].append(relative_error(frame, target))
        errors["shuffled"].append(relative_error(world_model(frame, z.roll(1, dims=0)), target))
        kls.append(kl)
    metrics = {name: torch.cat(values).median().item() for name, values in errors.items()}
    return {**metrics, "kl": torch.stack(kls).mean().item()}


def config_snapshot():
    """Return the uppercase constants of config.py as strings, stored in checkpoints to spot config changes on resume."""
    return {key: str(value) for key, value in vars(config).items() if key.isupper()}


def save_checkpoint(path, model, optimizer, epoch, step, best, val_generator, device):
    """Write everything needed to resume: weights, optimizer state, progress counters, RNG states and the config snapshot."""
    state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "step": step,
        "best": best,
        "rng": torch.get_rng_state(),
        "val_rng": val_generator.get_state(),
        "config": config_snapshot(),
    }
    if device == "mps":
        state["mps_rng"] = torch.mps.get_rng_state()
    tmp = path.with_suffix(".tmp")
    torch.save(state, tmp)
    os.replace(tmp, path)


def load_checkpoint(path, model, optimizer, val_generator, device):
    """Restore a checkpoint written by save_checkpoint and return (next epoch, step, best validation error)."""
    state = torch.load(path, map_location="cpu")
    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    torch.set_rng_state(state["rng"])
    val_generator.set_state(state["val_rng"])
    if device == "mps" and "mps_rng" in state:
        torch.mps.set_rng_state(state["mps_rng"])
    changed = [key for key, value in config_snapshot().items() if state["config"].get(key) != value]
    if changed:
        print(f"warning: config differs from the checkpoint in {changed}")
    return state["epoch"] + 1, state["step"], state["best"]


def main():
    """Train the encoder and world model on the train split, evaluating on the val split after every epoch."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="run")
    parser.add_argument("--epochs", type=int, default=EPOCHS, help="total number of epochs to reach, including resumed ones")
    parser.add_argument("--resume", action="store_true", help="continue from last.pt in the run folder")
    args = parser.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    torch.manual_seed(SEED)
    run_dir = ROOT / "artifacts" / "runs" / args.name
    run_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(run_dir / "tb", flush_secs=5)

    encoder, world_model = LatentActionEncoder().to(device), WorldModel().to(device)
    model = nn.ModuleDict({"encoder": encoder, "world_model": world_model})
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    diagnostics = Diagnostics(model)
    val_generator = torch.Generator().manual_seed(SEED)
    train_loader = DataLoader(PairDataset("train"), batch_size=TRAIN_BATCH_SIZE, shuffle=True, drop_last=True)
    val_loader = DataLoader(PairDataset("val"), batch_size=TRAIN_BATCH_SIZE, shuffle=True, generator=val_generator)

    start_epoch, step, best = 0, 0, float("inf")
    if args.resume:
        start_epoch, step, best = load_checkpoint(run_dir / "last.pt", model, optimizer, val_generator, device)
        print(f"resumed at epoch {start_epoch}, training to epoch {args.epochs}")

    for epoch in range(start_epoch, args.epochs):
        start = time.time()
        model.train()
        total, norms, clipped = (torch.zeros((), device=device) for _ in range(3))
        for batch in train_loader:
            diagnostics.active = step % LOG_EVERY == 0
            pred, _, target, _, kl = run_model(encoder, world_model, batch.to(device), sample=True)
            mse = F.mse_loss(pred, target)
            loss = mse + BETA * kl
            optimizer.zero_grad()
            loss.backward()
            if diagnostics.active:
                diagnostics.log(model, step, run_dir / "diagnostics.jsonl")
                diagnostics.active = False
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            total += loss.detach()
            norms += grad_norm
            clipped += grad_norm > GRAD_CLIP
            if step % SCALAR_EVERY == 0:
                writer.add_scalar("train/loss", loss.item(), step)
                writer.add_scalar("train/mse", mse.item(), step)
                writer.add_scalar("train/kl", kl.item(), step)
            step += 1
        metrics = evaluate(encoder, world_model, val_loader, device)
        n = len(train_loader)
        train = {"train_loss": total.item() / n, "grad_norm": norms.item() / n, "clipped": clipped.item() / n}
        print(
            f"epoch {epoch}: {time.time() - start:.0f}s train loss {train['train_loss']:.4f} | val error model "
            f"{metrics['model']:.3f} copy {metrics['copy']:.3f} shuffled {metrics['shuffled']:.3f} | kl {metrics['kl']:.2f} "
            f"| grad norm {train['grad_norm']:.3f} clipped {train['clipped']:.0%}"
        )
        with open(run_dir / "metrics.jsonl", "a") as f:
            f.write(json.dumps({"epoch": epoch, **train, **metrics}) + "\n")
        for name, value in train.items():
            writer.add_scalar(f"epoch/{name}", value, step)
        for name, value in metrics.items():
            writer.add_scalar(f"val/{name}", value, step)
        writer.flush()
        improved = metrics["model"] < best
        best = min(best, metrics["model"])
        save_checkpoint(run_dir / "last.pt", model, optimizer, epoch, step, best, val_generator, device)
        if improved:
            save_checkpoint(run_dir / "best.pt", model, optimizer, epoch, step, best, val_generator, device)
    writer.close()


if __name__ == "__main__":
    main()
