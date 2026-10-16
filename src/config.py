"""Hyperparameters shared across stages."""
import os
from pathlib import Path

import torch

ROOT = Path(__file__).parent.parent


def _env(name, default):
    """Return the environment override of a config constant, cast to the default's type, for run sweeps."""
    return type(default)(os.environ.get(name, default))


DATA_DIR = ROOT / "data"
ARTIFACT_DIR = ROOT / "artifacts"
FIGURE_DIR = ROOT / "docs" / "figures"
VIDEO_DIR = DATA_DIR / "human_videos"
DEMO_DIR = DATA_DIR / "panda_demos"
PLAY_DIR = DATA_DIR / "panda_play"
ROBOTIQ_DIR = DATA_DIR / "robotiq_play"
FEATURE_DIR = ARTIFACT_DIR / "features"
CHECKPOINT_DIR = ARTIFACT_DIR / "checkpoints"
SIM_STRIDE = 4  # control steps (20 Hz) between sim frames: 0.2 s, the same gap as 12 frames at 60 fps in the human clips
ACTION_DIM = 7  # normalised end-effector delta (3 position, 3 rotation) and gripper command
MODEL_NAME = "facebook/dinov3-vits16-pretrain-lvd1689m"
IMG_SIZE = (224, 384)
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
BATCH_SIZE = 32

STRIDE = 12
GRID = (14, 24)
POOL = 1
N_TOKENS = (GRID[0] // POOL) * (GRID[1] // POOL)
FEATURE_DIM = 384
D_MODEL = 256
N_HEADS = 4
ENC_LAYERS = _env("ENC_LAYERS", 2)
DEC_LAYERS = _env("DEC_LAYERS", 2)
N_QUERIES = _env("N_QUERIES", 2)
LATENT_DIM = _env("LATENT_DIM", 8)
BETA = _env("BETA", 0.0)
DROPOUT = _env("DROPOUT", 0.0)  # in every transformer layer and the decoder cross-attention
SAMPLE_LATENT = bool(_env("SAMPLE_LATENT", 0))  # set to 1 in the environment; True adds VAE noise to z (use with BETA > 0); False keeps z deterministic
CHUNK = 5  # future frames predicted from F_0; CHUNK = 1 is the single-step model
SEED = 0
VAL_FRACTION = 0.2

LR = 3e-4
WEIGHT_DECAY = _env("WEIGHT_DECAY", 0.01)
KL_WARMUP = _env("KL_WARMUP", 0)  # epochs over which beta ramps linearly from 0 to BETA; 0 disables the ramp
EMA_DECAY = _env("EMA_DECAY", 0.0)  # 0 turns the weight average off; evaluation and best.pt use the averaged weights when on
GRAD_CLIP = 1.0
EPOCHS = 20
TRAIN_BATCH_SIZE = 32
EVAL_BATCH_SIZE = 256
LOG_EVERY = 100
SCALAR_EVERY = 10


def get_device():
    """Pick cuda, then mps, then cpu."""
    return "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
