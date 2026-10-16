"""Shared figure style: the palette, Title Case for every piece of text, and an optional caption line."""
import re

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")

SURFACE, INK, MUTED = "#fcfcfb", "#0b0b0b", "#52514e"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"  # categorical slots 1 to 3, always assigned in this order
GREEN = (30, 200, 80)  # the goal silhouette


def capitalise(text):
    """Return the text with the first letter of every word in capitals, leaving the rest as written (so "t-SNE" becomes "T-SNE" and "window's" stays "Window's")."""
    return re.sub(r"(?<![\w'])[a-z]", lambda m: m.group().upper(), text)


def overlay_silhouette(frames, mask, colour=GREEN, alpha=0.45):
    """Return the RGB frames (N, H, W, 3) uint8 with the mask region tinted and outlined in `colour`."""
    inside = mask[None, ..., None]
    tinted = np.where(inside, (1 - alpha) * frames + alpha * np.array(colour), frames)
    edge = mask & ~(np.roll(mask, 1, 0) & np.roll(mask, -1, 0) & np.roll(mask, 1, 1) & np.roll(mask, -1, 1))
    return np.where(edge[None, ..., None], np.array(colour), tinted).astype(np.uint8)


def new_figure(rows=1, cols=1, size=(7, 4.4)):
    """Return a figure and its axes grid (always 2-D) on the shared surface colour."""
    fig, axes = plt.subplots(rows, cols, figsize=size, facecolor=SURFACE, squeeze=False)
    return fig, axes


def style_axes(ax, title=None, xlabel=None, ylabel=None):
    """Apply the shared look to one axes and set its title and axis labels in Title Case."""
    ax.set_facecolor(SURFACE)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(colors=INK)
    if title:
        ax.set_title(capitalise(title), color=INK)
    if xlabel:
        ax.set_xlabel(capitalise(xlabel), color=MUTED)
    if ylabel:
        ax.set_ylabel(capitalise(ylabel), color=MUTED)


def finish(fig, path, caption=None, dpi=150):
    """Add the caption in Title Case under the figure if given, save it to `path` and close it."""
    fig.tight_layout(rect=(0, 0.05 if caption else 0, 1, 1))
    if caption:
        fig.text(0.5, 0.012, capitalise(caption), ha="center", color=MUTED, fontsize=9)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
