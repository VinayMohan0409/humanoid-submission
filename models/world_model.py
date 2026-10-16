"""Single-step world model: predicts F_{t+h} from F_t and a latent action z."""
import torch
from torch import nn

from config import CONDITIONING, D_MODEL, DEC_LAYERS, FEATURE_DIM, LATENT_DIM, N_HEADS, N_QUERIES, N_TOKENS
from models.layers import transformer


class WorldModel(nn.Module):
    """Decoder that injects the latent action by cross-attention or by concatenation, then mixes tokens with self-attention."""

    def __init__(self):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(FEATURE_DIM, D_MODEL), nn.LayerNorm(D_MODEL))
        self.pos = nn.Parameter(torch.zeros(N_TOKENS, D_MODEL))
        if CONDITIONING == "cross":
            self.action_proj = nn.Linear(LATENT_DIM, D_MODEL)
            self.cross_norm = nn.LayerNorm(D_MODEL)
            self.cross = nn.MultiheadAttention(D_MODEL, N_HEADS, batch_first=True)
        else:
            self.action_proj = nn.Linear(N_QUERIES * LATENT_DIM, D_MODEL)
            self.fuse = nn.Linear(2 * D_MODEL, D_MODEL)
        self.blocks = transformer(DEC_LAYERS)
        self.norm = nn.LayerNorm(D_MODEL)
        self.out = nn.Linear(D_MODEL, FEATURE_DIM)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def condition(self, x, z):
        """Mix the latent action z into the frame tokens x, by cross-attention or by concatenating it to every token."""
        if CONDITIONING == "cross":
            action = self.action_proj(z)
            return x + self.cross(self.cross_norm(x), action, action, need_weights=False)[0]
        action = self.action_proj(z.flatten(1)).unsqueeze(1).expand(-1, x.shape[1], -1)
        return self.fuse(torch.cat([x, action], dim=-1))

    def forward(self, frame, z):
        """Return the predicted next frame: the input tokens plus a change computed from them and the action."""
        x = self.condition(self.proj(frame) + self.pos, z)
        return frame + self.out(self.norm(self.blocks(x)))
