import cv2
import numpy as np
import pytest

from config import FEATURE_DIR, VIDEO_DIR


def test_cache_matches_video_frames():
    cached = list(FEATURE_DIR.glob("*.npy"))
    if not cached:
        pytest.skip("run python -m features.dino first")
    for path in cached:
        n_frames = int(cv2.VideoCapture(str(VIDEO_DIR / f"{path.stem}.mp4")).get(cv2.CAP_PROP_FRAME_COUNT))
        assert np.load(path, mmap_mode="r").shape[0] == n_frames
