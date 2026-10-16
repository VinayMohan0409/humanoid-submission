"""Shared transformer building block."""
from torch import nn

from config import D_MODEL, N_HEADS


def transformer(n_layers):
    """Build a pre-norm stack of self-attention blocks with GELU MLPs at width D_MODEL."""
    layer = nn.TransformerEncoderLayer(
        D_MODEL, N_HEADS, 4 * D_MODEL, dropout=0.0, activation=nn.GELU(), batch_first=True, norm_first=True
    )
    return nn.TransformerEncoder(layer, n_layers, enable_nested_tensor=False)
