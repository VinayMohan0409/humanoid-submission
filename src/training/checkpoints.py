"""Save and restore training state so a run can resume."""
import os

import torch

from src import config
from src.models.latent_action import LatentActionEncoder
from src.models.world_model import WorldModel


def config_snapshot():
    """Return the uppercase constants of config.py as strings, stored in checkpoints to spot config changes on resume."""
    return {key: str(value) for key, value in vars(config).items() if key.isupper()}


def save_checkpoint(path, model, optimizer, epoch, step, best, val_generator, device, ema=None):
    """Write everything needed to resume: weights, optimizer state, progress counters, RNG states and the config snapshot."""
    state = {
        "model": ema.module.state_dict() if ema else model.state_dict(),
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
    elif device == "cuda":
        state["cuda_rng"] = torch.cuda.get_rng_state()
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
    elif device == "cuda" and "cuda_rng" in state:
        torch.cuda.set_rng_state(state["cuda_rng"])
    changed = [key for key, value in config_snapshot().items() if state["config"].get(key) != value]
    if changed:
        print(f"warning: config differs from the checkpoint in {changed}")
    return state["epoch"] + 1, state["step"], state["best"]


def load_model(path, device):
    """Return the encoder and world model of a co-trained or pretrained checkpoint, in eval mode."""
    model = torch.nn.ModuleDict({"encoder": LatentActionEncoder(), "world_model": WorldModel()}).to(device).eval()
    model.load_state_dict(torch.load(path, map_location=device)["model"])
    return model
