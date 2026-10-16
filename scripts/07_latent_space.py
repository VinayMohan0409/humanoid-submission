"""Latent space of the co-trained model: PCA, t-SNE and UMAP of the latent actions z of held-out human and Panda windows, coloured by embodiment."""
import argparse
from pathlib import Path

import numpy as np
import torch
import umap
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

from src.config import CHECKPOINT_DIR, FIGURE_DIR, LATENT_DIM, N_QUERIES, SEED, get_device
from src.datasets.human import ChunkDataset
from src.datasets.sim import SimFrames
from src.models.latent_action import encode_chunk
from src.plotting import BLUE, ORANGE, finish, new_figure, style_axes
from src.training.checkpoints import load_model

WINDOWS = 600  # held-out windows per embodiment, evenly spaced so neighbours barely overlap
EMBODIMENTS = {"Human": (BLUE, ChunkDataset), "Panda": (ORANGE, SimFrames)}
METHODS = {
    "PCA": lambda: PCA(2, random_state=SEED),
    "t-SNE": lambda: TSNE(2, perplexity=30, init="pca", random_state=SEED),
    "UMAP": lambda: umap.UMAP(n_neighbors=30, min_dist=0.3, init="random", random_state=SEED),
}


@torch.no_grad()
def collect(model, dataset):
    """Return the latent actions of evenly spaced held-out windows, one point per predicted step: (WINDOWS * CHUNK, N_QUERIES * LATENT_DIM)."""
    actions = []
    for batch in dataset.batches(256, shuffle=False):
        with torch.autocast(batch.device.type, torch.bfloat16, enabled=batch.is_cuda):
            mean, _ = encode_chunk(model["encoder"], batch)
        actions.append(mean.float().flatten(2).cpu())
    keep = np.linspace(0, len(dataset) - 1, WINDOWS).astype(int)
    return torch.cat(actions)[keep].flatten(0, 1).numpy()


def save_plot(path, points, labels):
    """Draw one panel per method with the points coloured by embodiment."""
    fig, axes = new_figure(1, len(METHODS), size=(12, 4.4))
    for ax, (method, build) in zip(axes[0], METHODS.items()):
        xy = build().fit_transform(points)
        for label, (name, (colour, _)) in enumerate(EMBODIMENTS.items()):
            ax.scatter(*xy[labels == label].T, s=5, alpha=0.5, color=colour, label=name, linewidths=0)
        ax.set(xticks=[], yticks=[])
        style_axes(ax, title=method)
    axes[0, 0].legend(frameon=False, markerscale=3)
    fig.suptitle(f"Latent Actions Coloured By Embodiment ({N_QUERIES} Tokens × {LATENT_DIM} Dimensions)")
    finish(fig, path)


def main():
    """Embed held-out human and Panda latent actions and write latent_space.png."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=str(CHECKPOINT_DIR / "cotrain" / "last.pt"))
    args = parser.parse_args()
    device = get_device()
    model = load_model(args.checkpoint, device)
    per_embodiment = [collect(model, dataset_class("val", device)) for _, dataset_class in EMBODIMENTS.values()]
    points = StandardScaler().fit_transform(np.concatenate(per_embodiment))
    labels = np.concatenate([np.full(len(p), i) for i, p in enumerate(per_embodiment)])
    path = FIGURE_DIR / f"latent_space_{Path(args.checkpoint).parent.name}.png"
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    save_plot(path, points, labels)
    print(path)


if __name__ == "__main__":
    main()
