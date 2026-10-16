"""Torch dataset of (F_t, F_{t+h}) feature pairs from the cached DINO tokens."""
import csv
import random
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import Dataset

from config import FEATURE_DIR, GRID, POOL, SEED, STRIDE, VAL_FRACTION, VIDEO_DIR


def read_split(split):
    """Return the clip names of "train" or "val": a seeded random 20% of each (object, drop) group, at least one clip, goes to val."""
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
    return [name for name in names if (name in val) == (split == "val")]


def pool(tokens):
    """Average-pool the token grid by POOL; with POOL = 1 the tokens come back unchanged."""
    if POOL == 1:
        return tokens
    grid = tokens.view(-1, *GRID, tokens.shape[-1]).permute(0, 3, 1, 2)
    return torch.nn.functional.avg_pool2d(grid, POOL).flatten(2).transpose(1, 2)


class PairDataset(Dataset):
    """Every (F_t, F_{t+STRIDE}) pair from the clips of one split, read lazily from the memory-mapped cache."""

    def __init__(self, split):
        self.clips = [np.load(FEATURE_DIR / f"{name}.npy", mmap_mode="r") for name in read_split(split)]
        self.index = [(c, t) for c, clip in enumerate(self.clips) for t in range(len(clip) - STRIDE)]

    def __len__(self):
        """Return the number of pairs across all clips in the split."""
        return len(self.index)

    def __getitem__(self, i):
        """Return a float32 tensor of shape (2, tokens, 384) holding F_t and F_{t+STRIDE}."""
        c, t = self.index[i]
        pair = torch.from_numpy(np.stack([self.clips[c][t], self.clips[c][t + STRIDE]])).float()
        return pool(pair)
