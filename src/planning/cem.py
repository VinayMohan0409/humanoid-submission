"""CEM planner over latent actions."""
import torch
from torch.nn import functional as F

from src.config import CHUNK


def fit_prior(z):
    """Fit a diagonal Gaussian to the latent actions z (..., Q, L) seen in the demonstrations; return its mean and std per latent number, each (Q, L)."""
    flat = z.flatten(0, z.dim() - 3)
    return flat.mean(0), flat.std(0)


def token_distance(predicted, goal, metric):
    """Return the per-token distance (..., tokens) between predicted and goal features: squared error ("l2"), one minus cosine similarity ("cosine"), or the two each divided by its mean over the candidates and added ("both")."""
    if metric == "l2":
        return (predicted - goal).pow(2).mean(-1)
    cosine = 1 - (F.normalize(predicted, dim=-1) * F.normalize(goal, dim=-1)).sum(-1)
    if metric == "cosine":
        return cosine
    l2 = (predicted - goal).pow(2).mean(-1)
    return l2 / l2.mean() + cosine / cosine.mean()


@torch.no_grad()
def cem(world_model, frame, goal, prior, samples=256, elites=32, iters=4, weights=None, step_weights=None, metric="l2"):
    """Return the latent action sequence (CHUNK, Q, L) whose predicted frames come closest to the goal features, and its best distance; `weights` (tokens,) weights each token's error, uniform by default, and `step_weights` (CHUNK,) weights each predicted frame's error, defaulting to the best-matching frame alone; `metric` is the per-token distance: "l2", "cosine" or "both" (see token_distance)."""
    weights = torch.full((frame.shape[1],), 1 / frame.shape[1], device=frame.device) if weights is None else weights
    mean, std = (p.expand(CHUNK, -1, -1).clone() for p in prior)
    for _ in range(iters):
        z = mean + std * torch.randn(samples, *mean.shape, device=frame.device)
        with torch.autocast(frame.device.type, torch.bfloat16, enabled=frame.is_cuda):
            predicted = world_model(frame.expand(samples, -1, -1), z).float()
        per_frame = (token_distance(predicted, goal, metric) * weights).sum(2)  # (samples, CHUNK)
        cost = per_frame.min(dim=1).values if step_weights is None else (per_frame * step_weights).sum(1)
        elite = z[cost.topk(elites, largest=False).indices]
        mean, std = elite.mean(0), elite.std(0) + 1e-3
    return mean, cost.min().item()
