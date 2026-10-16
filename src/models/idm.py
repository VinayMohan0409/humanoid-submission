"""Inverse dynamics model: the actions that take one frame's DINO features to the next's."""
import torch
from torch import nn

from src.config import ENC_LAYERS, FEATURE_DIM, N_HEADS, N_TOKENS, SIM_STRIDE


class InverseDynamics(nn.Module):
    """Maps the DINO features of two frames SIM_STRIDE steps apart to the actions executed between them: (B, SIM_STRIDE, 3) positions, or (B, SIM_STRIDE, 4) with the gripper command last."""

    def __init__(self, d_model=256, dropout=0.0, layers=ENC_LAYERS, head_hidden=0, gripper=False):
        super().__init__()
        self.outputs = 4 if gripper else 3
        self.proj = nn.Sequential(nn.Linear(FEATURE_DIM, d_model), nn.LayerNorm(d_model))
        self.pos = nn.Parameter(torch.zeros(N_TOKENS, d_model))
        self.frame_emb = nn.Parameter(torch.zeros(2, d_model))
        self.query = nn.Parameter(torch.randn(1, d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(d_model, N_HEADS, 4 * d_model, dropout=dropout, activation=nn.GELU(), batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_model)
        out = SIM_STRIDE * self.outputs
        self.head = nn.Sequential(nn.Linear(d_model, head_hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(head_hidden, out)) if head_hidden else nn.Linear(d_model, out)

    def forward(self, frame, next_frame):
        """Return the actions that took `frame` to `next_frame`."""
        tokens = [self.proj(f) + self.pos + self.frame_emb[i] for i, f in enumerate((frame, next_frame))]
        x = torch.cat([*tokens, self.query.expand(len(frame), -1, -1)], dim=1)
        return self.head(self.norm(self.transformer(x)[:, -1])).view(-1, SIM_STRIDE, self.outputs)


def action_columns(outputs):
    """Return the columns of the 7-d robot action the IDM predicts: x, y, z, and with 4 outputs the gripper command."""
    return [0, 1, 2, 6][:outputs]


def pairs(batch, outputs):
    """Flatten a window batch into consecutive frame pairs and their actions: (N, tokens, 384), (N, tokens, 384), (N, SIM_STRIDE, outputs)."""
    frames, actions, _ = batch
    return frames[:, :-1].flatten(0, 1), frames[:, 1:].flatten(0, 1), actions[..., action_columns(outputs)].flatten(0, 1)
