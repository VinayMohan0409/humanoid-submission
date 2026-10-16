import csv

import numpy as np
import pytest
import torch

from config import FEATURE_DIR, STRIDE, VIDEO_DIR
from data.dataset import PairDataset, pool, read_split


def test_split_is_disjoint_and_complete():
    if not (VIDEO_DIR / "metadata.csv").exists():
        pytest.skip("dataset not downloaded")
    with open(VIDEO_DIR / "metadata.csv") as f:
        names = {row["file_name"].removesuffix(".mp4") for row in csv.DictReader(f)}
    train, val = set(read_split("train")), set(read_split("val"))
    assert train.isdisjoint(val)
    assert train | val == names


def test_pair_is_stride_apart():
    if not list(FEATURE_DIR.glob("*.npy")):
        pytest.skip("run python -m features.dino first")
    dataset = PairDataset("train")
    clip = dataset.clips[0]
    expected = pool(torch.from_numpy(np.stack([clip[0], clip[STRIDE]])).float())
    assert torch.equal(dataset[0], expected)
