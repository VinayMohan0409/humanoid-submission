"""Check the human-video train/val split and the stride between windowed frames."""
import csv

import numpy as np
import pytest
import torch

from src.config import CHUNK, FEATURE_DIR, STRIDE, VIDEO_DIR
from src.datasets.human import ChunkDataset, pool, read_split


def test_split_is_disjoint_and_complete():
    if not (VIDEO_DIR / "metadata.csv").exists():
        pytest.skip("dataset not downloaded")
    with open(VIDEO_DIR / "metadata.csv") as f:
        names = {row["file_name"].removesuffix(".mp4") for row in csv.DictReader(f)}
    train, val = set(read_split("train")), set(read_split("val"))
    assert train.isdisjoint(val)
    assert train | val == names


def test_window_is_stride_apart():
    if not list(FEATURE_DIR.glob("*.npy")):
        pytest.skip("run python -m scripts.01_extract_features human_videos first")
    dataset = ChunkDataset("train", "cpu")
    clip = np.load(FEATURE_DIR / f"{read_split('train')[0]}.npy")
    expected = pool(torch.from_numpy(np.stack([clip[k * STRIDE] for k in range(CHUNK + 1)])).float())
    first = next(dataset.batches(1, shuffle=False))[0]
    assert torch.equal(first, expected)
