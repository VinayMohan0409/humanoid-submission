"""Frame-feature windows from the cached DINO tokens, served from GPU memory."""
import csv
import random
from collections import defaultdict

import numpy as np
import torch

from src.config import CHUNK, FEATURE_DIR, GRID, POOL, SEED, STRIDE, VAL_FRACTION, VIDEO_DIR


def read_split(split):
    """Return the clip names of "train", "val" or "all": a seeded random 20% of each (object, drop) group, at least one clip, goes to val."""
    with open(VIDEO_DIR / "metadata.csv") as f:
        groups = defaultdict(list)
        for row in csv.DictReader(f):
            groups[(row["object"], row["drop"])].append(row["file_name"].removesuffix(".mp4"))
    rng = random.Random(SEED)
    val = set()
    for key in sorted(groups):
        clips = sorted(groups[key])
        val.update(rng.sample(clips, max(1, round(VAL_FRACTION * len(clips)))))
    names = sorted(name for group in groups.values() for name in group)
    return [name for name in names if split == "all" or (name in val) == (split == "val")]


def pool(tokens):
    """Average-pool the token grid by POOL; with POOL = 1 the tokens come back unchanged."""
    if POOL == 1:
        return tokens
    grid = tokens.view(-1, *GRID, tokens.shape[-1]).permute(0, 3, 1, 2)
    return torch.nn.functional.avg_pool2d(grid, POOL).flatten(2).transpose(1, 2)


class ChunkDataset:
    """Every window (F_t, F_{t+S}, ..., F_{t+CHUNK*S}) from the clips of one split, with all features resident on `device`."""

    def __init__(self, split, device):
        clips = [torch.from_numpy(np.load(FEATURE_DIR / f"{name}.npy")) for name in read_split(split)]
        self.frames = torch.cat(clips).to(device)
        starts = np.cumsum([0] + [len(clip) for clip in clips])
        self.starts = torch.tensor([s + t for s, n in zip(starts, map(len, clips)) for t in range(n - CHUNK * STRIDE)], device=device)
        self.offsets = torch.arange(CHUNK + 1, device=device) * STRIDE

    def __len__(self):
        """Return the number of windows across all clips in the split."""
        return len(self.starts)

    def batches(self, batch_size, shuffle, generator=None, drop_last=False):
        """Yield float32 tensors of shape (B, CHUNK + 1, tokens, 384) holding frames STRIDE apart."""
        order = torch.randperm(len(self), generator=generator) if shuffle else torch.arange(len(self))
        stop = len(self) - len(self) % batch_size if drop_last else len(self)
        for i in range(0, stop, batch_size):
            idx = self.starts[order[i : i + batch_size].to(self.starts.device)]
            window = self.frames[idx[:, None] + self.offsets].float()
            yield pool(window.flatten(0, 1)).view(len(idx), CHUNK + 1, -1, window.shape[-1])
