"""Chunk world model: predicts F_1..F_K from F_0 and K latent actions."""
import torch
from torch import nn

from src import config
from src.config import CHUNK, D_MODEL, DEC_LAYERS, FEATURE_DIM, LATENT_DIM, N_HEADS, N_QUERIES, N_TOKENS
from src.models.layers import transformer


class WorldModel(nn.Module):
    """Block-causal decoder: predicted frame k cross-attends to latent actions 1..k only, then mixes tokens with self-attention."""

    def __init__(self):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(FEATURE_DIM, D_MODEL), nn.LayerNorm(D_MODEL))
        self.pos = nn.Parameter(torch.zeros(N_TOKENS, D_MODEL))
        self.step_emb = nn.Parameter(torch.zeros(CHUNK, 1, D_MODEL))
        self.action_emb = nn.Parameter(torch.zeros(CHUNK, 1, D_MODEL))
        self.action_proj = nn.Linear(LATENT_DIM, D_MODEL)
        self.cross_norm = nn.LayerNorm(D_MODEL)
        self.cross = nn.MultiheadAttention(D_MODEL, N_HEADS, dropout=config.DROPOUT, batch_first=True)
        self.blocks = transformer(DEC_LAYERS)
        self.norm = nn.LayerNorm(D_MODEL)
        self.out = nn.Linear(D_MODEL, FEATURE_DIM)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)
        later = torch.arange(CHUNK)[:, None] < torch.arange(CHUNK)[None, :]
        self.register_buffer("mask", later.repeat_interleave(N_TOKENS, 0).repeat_interleave(N_QUERIES, 1))

    def forward(self, frame, z):
        """Map F_0 (B, N, D) and actions z (B, K, Q, L) to predicted frames (B, K, N, D): F_0 plus a change computed from them."""
        b, k = z.shape[:2]
        x = ((self.proj(frame) + self.pos).unsqueeze(1) + self.step_emb).flatten(1, 2)
        action = (self.action_proj(z) + self.action_emb).flatten(1, 2)
        x = x + self.cross(self.cross_norm(x), action, action, attn_mask=self.mask, need_weights=False)[0]
        x = self.blocks(x.reshape(b * k, N_TOKENS, -1))
        return frame.unsqueeze(1) + self.out(self.norm(x)).view(b, k, N_TOKENS, FEATURE_DIM)
