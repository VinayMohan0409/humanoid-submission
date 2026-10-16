"""Forward pass and validation metrics for the latent action world model."""
import torch

from src.config import EVAL_BATCH_SIZE, SAMPLE_LATENT
from src.models.latent_action import encode_chunk, kl_divergence, sample_latent


def squared_error(pred, target):
    """Return each frame's summed squared error over tokens and features, shape (B, K)."""
    return (pred - target).flatten(2).pow(2).sum(dim=2)


def run_model(encoder, world_model, batch, sample):
    """Encode consecutive frame pairs into K latent actions z, then predict F_1..F_K from F_0 and z; returns prediction, F_0, targets, z and KL."""
    frame, target = batch[:, 0], batch[:, 1:]
    with torch.autocast(batch.device.type, torch.bfloat16, enabled=batch.is_cuda):
        mean, logvar = encode_chunk(encoder, batch)
        z = sample_latent(mean, logvar) if sample and SAMPLE_LATENT else mean
        pred = world_model(frame, z)
    return pred.float(), frame, target, z.float(), kl_divergence(mean.float(), logvar.float())


@torch.no_grad()
def evaluate(encoder, world_model, data, generator, device):
    """Return per-step MSE of the model and of the copy-F_0 baseline, their ratio for the model and shuffled latents (1 = no better than copy), and mean KL."""
    encoder.eval()
    world_model.eval()
    sums = {name: 0 for name in ("model", "copy", "shuffled")}
    kls, count = [], 0
    for batch in data.batches(EVAL_BATCH_SIZE, shuffle=True, generator=generator):
        pred, frame, target, z, kl = run_model(encoder, world_model, batch, sample=False)
        with torch.autocast(device, torch.bfloat16, enabled=device == "cuda"):
            shuffled = world_model(frame, z.roll(1, dims=0)).float()
        sums["model"] += squared_error(pred, target).sum(0)
        sums["copy"] += squared_error(frame.unsqueeze(1).expand_as(target), target).sum(0)
        sums["shuffled"] += squared_error(shuffled, target).sum(0)
        kls.append(kl)
        count += len(batch) * target.shape[2] * target.shape[3]
    metrics = {name: (sums[name] / sums["copy"]).tolist() for name in ("model", "shuffled")}
    mse = {"mse": (sums["model"] / count).tolist(), "mse_copy": (sums["copy"] / count).tolist()}
    return {**metrics, **mse, "kl": torch.stack(kls).mean().item()}


@torch.no_grad()
def latent_stats(encoder, data):
    """Return per-dimension aggregate mean and std of z over the data, KL per dimension, and the count of active dimensions (posterior-mean variance above 0.01)."""
    encoder.eval()
    n, s_mu, s_mu2, s_var, s_kl = 0, 0, 0, 0, 0
    for batch in data.batches(EVAL_BATCH_SIZE, shuffle=False):
        mean, logvar = (x.float().flatten(2) for x in encode_chunk(encoder, batch))
        n += mean.shape[0] * mean.shape[1]
        s_mu, s_mu2, s_var = s_mu + mean.sum((0, 1)), s_mu2 + mean.pow(2).sum((0, 1)), s_var + logvar.exp().sum((0, 1))
        s_kl = s_kl + (0.5 * (mean.pow(2) + logvar.exp() - 1 - logvar)).sum((0, 1))
    mu = s_mu / n
    var_mu = s_mu2 / n - mu**2
    return {"z_mean": mu.tolist(), "z_std": (var_mu + s_var / n).sqrt().tolist(), "kl_per_dim": (s_kl / n).tolist(), "active_dims": int((var_mu > 0.01).sum())}


@torch.no_grad()
def mean_latent(encoder, data):
    """Return the mean latent action over the data per predicted step, query and latent number, shape (CHUNK, N_QUERIES, LATENT_DIM)."""
    encoder.eval()
    total, count = 0, 0
    for batch in data.batches(EVAL_BATCH_SIZE, shuffle=False):
        with torch.autocast(batch.device.type, torch.bfloat16, enabled=batch.is_cuda):
            mean, _ = encode_chunk(encoder, batch)
        total, count = total + mean.float().sum(0), count + len(batch)
    return total / count


@torch.no_grad()
def latent_ablation(encoder, world_model, data, constant, generator):
    """Return per-step error over the copy-F_0 baseline's (1 = no better than copy) with the window's own latent, another window's latent, and the same `constant` latent for every window."""
    encoder.eval()
    world_model.eval()
    sums = {name: 0 for name in ("own", "shuffled", "constant", "copy")}
    for batch in data.batches(EVAL_BATCH_SIZE, shuffle=True, generator=generator):
        pred, frame, target, z, _ = run_model(encoder, world_model, batch, sample=False)
        with torch.autocast(batch.device.type, torch.bfloat16, enabled=batch.is_cuda):
            shuffled = world_model(frame, z.roll(1, dims=0)).float()
            fixed = world_model(frame, constant.expand_as(z)).float()
        for name, predicted in (("own", pred), ("shuffled", shuffled), ("constant", fixed)):
            sums[name] += squared_error(predicted, target).sum(0)
        sums["copy"] += squared_error(frame.unsqueeze(1).expand_as(target), target).sum(0)
    return {name: (sums[name] / sums["copy"]).tolist() for name in ("own", "shuffled", "constant")}
