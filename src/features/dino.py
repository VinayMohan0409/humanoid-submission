"""Extract and cache frozen DINOv3 patch tokens."""
import json

import cv2
import numpy as np
import torch
from transformers import AutoModel

from src.config import BATCH_SIZE, FEATURE_DIR, IMG_SIZE, MEAN, MODEL_NAME, STD, VIDEO_DIR, get_device


def load_frames(path):
    """Return the video's frames as an RGB uint8 array (N, H, W, 3) resized to IMG_SIZE."""
    cap = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, bgr = cap.read()
        if not ok:
            return np.stack(frames)
        resized = cv2.resize(bgr, IMG_SIZE[::-1], interpolation=cv2.INTER_AREA)
        frames.append(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB))


def load_model(device):
    """Return the frozen DINOv3 model and the image mean and std tensors on `device`."""
    model = AutoModel.from_pretrained(MODEL_NAME).eval().to(device)
    mean = torch.tensor(MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(STD, device=device).view(1, 3, 1, 1)
    return model, mean, std


@torch.no_grad()
def encode(model, mean, std, frames, device):
    """Return the patch tokens (N, tokens, 384) of RGB frames as float16, register and class tokens dropped."""
    n_prefix = 1 + model.config.num_register_tokens
    out = []
    for i in range(0, len(frames), BATCH_SIZE):
        x = torch.from_numpy(frames[i : i + BATCH_SIZE]).to(device).permute(0, 3, 1, 2).float() / 255
        tokens = model(pixel_values=(x - mean) / std).last_hidden_state[:, n_prefix:]
        out.append(tokens.half().cpu())
    return torch.cat(out).numpy()


def extract_videos():
    """Encode every human video in VIDEO_DIR and cache its features in FEATURE_DIR."""
    device = get_device()
    model, mean, std = load_model(device)
    FEATURE_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(VIDEO_DIR.glob("*.mp4")):
        features = encode(model, mean, std, load_frames(path), device)
        np.save(FEATURE_DIR / f"{path.stem}.npy", features)
        print(f"{path.stem}: {features.shape}")
    (FEATURE_DIR / "meta.json").write_text(json.dumps({"model": MODEL_NAME, "img_size": IMG_SIZE}))

