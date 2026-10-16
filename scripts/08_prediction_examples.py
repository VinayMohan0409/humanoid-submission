"""Show one typical window per case (human/sim x train/held-out): 6 RGB frames, ground-truth DINO, predicted DINO and error, saved as PNGs."""
import argparse

import h5py
import numpy as np
import torch
from torch import nn

from src.config import CHECKPOINT_DIR, FIGURE_DIR, CHUNK, FEATURE_DIR, GRID, SIM_STRIDE, STRIDE, VIDEO_DIR, get_device
from src.features.dino import load_frames
from src.datasets.human import ChunkDataset, read_split
from src.datasets.sim import DEMOS, SimFrames, sim_split
from src.models.latent_action import LatentActionEncoder
from src.models.world_model import WorldModel
from src.plotting import capitalise, finish, new_figure, style_axes
from src.training.evaluation import run_model

DISPLAY = {"human": "Human", "sim": "Panda"}


@torch.no_grad()
def window_ratios(model, data):
    """Return every window's model squared error over the copy-F_0 baseline's, in dataset order."""
    ratios = []
    for batch in data.batches(256, shuffle=False):
        pred, frame, target, _, _ = run_model(model["encoder"], model["world_model"], batch, sample=False)
        ratios.append((pred - target).pow(2).sum((1, 2, 3)) / (frame.unsqueeze(1) - target).pow(2).sum((1, 2, 3)))
    return torch.cat(ratios)


def typical_window(ratios):
    """Return the index of the window whose ratio is closest to the median, and that median."""
    median = ratios.median()
    return int((ratios - median).abs().argmin()), median.item()


def locate_human(split, index):
    """Map a window index of the human split to (clip name, first frame) using the dataset's own window enumeration."""
    names = read_split(split)
    for name in names:
        n = len(np.load(FEATURE_DIR / f"{name}.npy", mmap_mode="r")) - CHUNK * STRIDE
        if index < n:
            return name, index
        index -= n


def locate_sim(split, index):
    """Map a window index of the sim split to (episode name, first frame)."""
    folder, every = DEMOS
    for name in sim_split(split, folder, every):
        with h5py.File(folder / "episodes" / f"{name}.h5") as f:
            n = len(f["action"]) - CHUNK * SIM_STRIDE + 1
        if index < n:
            return name, index
        index -= n


def rgb_frames(domain, name, start):
    """Return the CHUNK + 1 RGB frames of a window, as (6, H, W, 3) uint8."""
    if domain == "human":
        video = load_frames(VIDEO_DIR / f"{name}.mp4")
        return video[start + np.arange(CHUNK + 1) * STRIDE]
    with h5py.File(DEMOS[0] / "episodes" / f"{name}.h5") as f:
        return f["rgb"][start + np.arange(CHUNK + 1) * SIM_STRIDE]


def pca_colours(truth, pred):
    """Project both feature sets on the top 3 principal axes of the ground truth and scale to [0, 1] RGB maps of the token grid."""
    flat = truth.flatten(0, 1)
    mean = flat.mean(0)
    axes = torch.linalg.eigh(torch.cov((flat - mean).T))[1][:, -3:]
    project = lambda x: ((x - mean) @ axes).view(*x.shape[:-1], 3)
    t, p = project(truth), project(pred)
    low, high = (torch.quantile(t.flatten(0, 1), q, dim=0) for q in (0.02, 0.98))
    scale = lambda x: ((x - low) / (high - low)).clamp(0, 1).view(-1, *GRID, 3).cpu().numpy()
    return scale(t), scale(p)


def draw(path, title, rgb, truth, pred):
    """Save a 4 x 6 grid: RGB frames, ground-truth DINO (PCA colours), predicted DINO and per-patch squared error."""
    colours_t, colours_p = pca_colours(truth, pred)
    error = (pred - truth).pow(2).sum(-1).view(-1, *GRID).cpu().numpy()
    vmax = np.percentile(error[1:], 99)
    fig, axes = new_figure(4, CHUNK + 1, (15, 8.6))
    for k in range(CHUNK + 1):
        axes[0, k].imshow(rgb[k])
        axes[0, k].set_title(capitalise("F0 (input)" if k == 0 else f"F{k} (+{k * 0.2:.1f} seconds)"), fontsize=10)
        axes[1, k].imshow(colours_t[k], interpolation="nearest")
        axes[2, k].imshow(colours_p[k], interpolation="nearest")
        if k:
            axes[3, k].imshow(error[k], cmap="magma", vmin=0, vmax=vmax, interpolation="nearest")
    axes[3, 0].text(0.5, 0.5, "Input,\nNo Error", ha="center", va="center")
    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])
        style_axes(ax)
    for row, label in enumerate(("RGB frame", "ground-truth\nDINO", "predicted\nDINO", "squared error\nper patch")):
        style_axes(axes[row, 0], ylabel=label)
    fig.suptitle(capitalise(title))
    finish(fig, path, dpi=110)


def main():
    """Pick a typical window in each case, run the model on it and save the four figures."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default="cotrain")
    args = parser.parse_args()
    device = get_device()
    model = nn.ModuleDict({"encoder": LatentActionEncoder(), "world_model": WorldModel()}).to(device).eval()
    model.load_state_dict(torch.load(CHECKPOINT_DIR / args.run / "last.pt", map_location=device)["model"])
    out = FIGURE_DIR / "prediction_examples"
    out.mkdir(parents=True, exist_ok=True)
    for domain, make, locate in (("human", ChunkDataset, locate_human), ("sim", SimFrames, locate_sim)):
        for split, label in (("train", "train"), ("val", "held-out")):
            data = make(split, device)
            ratios = window_ratios(model, data)
            index, median = typical_window(ratios)
            name, start = locate(split, index)
            window = next(b[index % 256][None] for i, b in enumerate(data.batches(256, shuffle=False)) if i == index // 256)
            with torch.no_grad():
                pred, frame, target, _, _ = run_model(model["encoder"], model["world_model"], window, sample=False)
            truth = window[0]
            predicted = torch.cat([frame, pred[0]])
            title = f"{DISPLAY[domain]} {label} window: {name}, frame {start}  |  relative error {ratios[index]:.2f}, {label} median {median:.2f}"
            draw(out / f"{domain}_{label.replace('-', '')}.png", title, rgb_frames(domain, name, start), truth, predicted)
            print(title, flush=True)


if __name__ == "__main__":
    main()
