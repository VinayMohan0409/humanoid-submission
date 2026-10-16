"""Latent action encoder with a VAE bottleneck."""
import torch
from torch import nn

from src.config import D_MODEL, ENC_LAYERS, FEATURE_DIM, LATENT_DIM, N_QUERIES, N_TOKENS
from src.models.layers import transformer


class LatentActionEncoder(nn.Module):
    """Infers a latent action z from a pair of frame features (F_t, F_{t+h})."""

    def __init__(self):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(FEATURE_DIM, D_MODEL), nn.LayerNorm(D_MODEL))
        self.pos = nn.Parameter(torch.zeros(N_TOKENS, D_MODEL))
        self.frame_emb = nn.Parameter(torch.zeros(2, D_MODEL))
        self.queries = nn.Parameter(torch.randn(N_QUERIES, D_MODEL) * 0.02)
        self.transformer = transformer(ENC_LAYERS)
        self.norm = nn.LayerNorm(D_MODEL)
        self.head = nn.Linear(D_MODEL, 2 * LATENT_DIM)

    def forward(self, frame, next_frame):
        """Return (mean, logvar), each (batch, N_QUERIES, LATENT_DIM), describing the change frame -> next_frame."""
        tokens = [self.proj(f) + self.pos + self.frame_emb[i] for i, f in enumerate((frame, next_frame))]
        queries = self.queries.expand(len(frame), -1, -1)
        out = self.transformer(torch.cat([*tokens, queries], dim=1))[:, -N_QUERIES:]
        mean, logvar = self.head(self.norm(out)).chunk(2, dim=-1)
        return mean, logvar


def encode_chunk(encoder, frames):
    """Encode consecutive pairs of (B, K+1, N, D) frames into (mean, logvar), each (B, K, N_QUERIES, LATENT_DIM)."""
    b, k = frames.shape[0], frames.shape[1] - 1
    mean, logvar = encoder(frames[:, :-1].flatten(0, 1), frames[:, 1:].flatten(0, 1))
    return mean.view(b, k, *mean.shape[1:]), logvar.view(b, k, *logvar.shape[1:])


def sample_latent(mean, logvar):
    """Draw z = mean + std * noise, so gradients flow through mean and logvar."""
    return mean + torch.randn_like(mean) * (0.5 * logvar).exp()


def kl_divergence(mean, logvar):
    """KL from N(mean, std^2) to N(0, 1), summed over the latent numbers and averaged over the batch."""
    return (-0.5 * (1 + logvar - mean**2 - logvar.exp())).flatten(1).sum(1).mean()
