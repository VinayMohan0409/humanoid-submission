"""Extract and cache frozen DINOv3 patch tokens for simulated episodes."""
from pathlib import Path

import h5py
import numpy as np

from src.config import get_device
from src.features.dino import encode, load_model


def extract_episodes(folder):
    """Encode every episode's RGB frames in `folder` and save (frames, tokens, 384) features to its features subfolder."""
    device = get_device()
    model, mean, std = load_model(device)
    out = Path(folder) / "features"
    out.mkdir(exist_ok=True)
    for path in sorted((Path(folder) / "episodes").glob("ep_*.h5")):
        with h5py.File(path) as f:
            features = encode(model, mean, std, f["rgb"][:], device)
        np.save(out / f"{path.stem}.npy", features)
        print(f"{path.stem}: {features.shape}")
