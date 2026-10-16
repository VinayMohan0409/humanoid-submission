"""Hyperparameters shared across stages."""
from pathlib import Path

ROOT = Path(__file__).parent
VIDEO_DIR = ROOT.parent / "humanoid-data"
FEATURE_DIR = ROOT / "artifacts" / "features"
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
ENC_LAYERS = 2
DEC_LAYERS = 2
N_QUERIES = 4
LATENT_DIM = 8
BETA = 0.0
CONDITIONING = "cross"  # or "concat"
SEED = 0
VAL_FRACTION = 0.2

LR = 3e-4
WEIGHT_DECAY = 0.01
GRAD_CLIP = 1.0
EPOCHS = 20
TRAIN_BATCH_SIZE = 64
LOG_EVERY = 100
SCALAR_EVERY = 10
