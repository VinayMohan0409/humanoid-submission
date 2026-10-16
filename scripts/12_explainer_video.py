"""Render the explainer video: data, DINOv3 latent states, the latent action encoder and causal decoder (with real predictions from the co-trained model), the IDM, planning, and task rollouts. Writes docs/videos/explainer.mp4, or preview frames with --preview."""
import argparse

import h5py
import imageio
import matplotlib
import numpy as np
import torch
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
from PIL import Image

from src.config import ARTIFACT_DIR, CHECKPOINT_DIR, DEMO_DIR, FEATURE_DIR, FIGURE_DIR, GRID, PLAY_DIR, SIM_STRIDE, STRIDE, VIDEO_DIR
from src.plotting import capitalise
from src.training.checkpoints import load_model
from src.training.evaluation import run_model

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

matplotlib.rcParams.update({"font.family": "Liberation Sans", "mathtext.fontset": "custom", "mathtext.rm": "Liberation Sans", "mathtext.it": "Liberation Sans:italic", "mathtext.bf": "Liberation Sans:bold"})
FPS = 20
BG, INK, MUTED = "#14161a", "#f0efec", "#a9abb3"
SLATE, SAGE, CLAY, SAND, STONE = "#8FA3BF", "#A9C0AE", "#D4A98F", "#E0CB9A", "#C9CCD1"
HUMAN_CLIPS = {"cards_03": "Cards", "cube_05": "Cube", "cup_07": "Cup", "cylinder_04": "Cylinder", "duster_06": "Duster", "cube_drop_01": "Cube (Dropped)"}
DEMO_CLIPS = {"ep_001": "Placed", "ep_002": "Placed", "ep_000": "Released Mid-Carry", "ep_005": "Edge Grasp", "ep_009": "Missed Grasp", "ep_012": "Placed"}
ROLLOUTS = [("cotrain_panda", "reach", 3, "Co-Trained · Panda Gripper · Reach"), ("cotrain_panda", "carry", 1, "Co-Trained · Panda Gripper · Carry A Cuboid"), ("cotrain_robotiq", "reach", 3, "Co-Trained · Robotiq Gripper · Reach")]
HUMAN_WINDOW, PANDA_WINDOW = ("cup_05", 38), ("ep_024", 55)  # the held-out windows shown in the prediction examples
OUT = FIGURE_DIR.parent / "videos"
COL_X = [0.7 + k * 2.5 for k in range(6)]  # the six frame columns shared by the latent state and world model scenes
COL_W = 2.2


def ease(t, start, duration=0.6):
    """Return a smooth 0 -> 1 ramp that starts at `start` and lasts `duration` seconds."""
    x = min(max((t - start) / duration, 0.0), 1.0)
    return x * x * (3 - 2 * x)


def pulse(t, start, hold=1.2, duration=0.4):
    """Return a ramp up at `start`, a hold, then a ramp down: 0 -> 1 -> 0."""
    return ease(t, start, duration) * (1 - ease(t, start + duration + hold, duration))


def read_video(path, every=1, size=None, limit=None):
    """Return a video's frames as a uint8 array, keeping every `every`-th frame, resized to `size` (width, height), at most `limit` frames."""
    frames = []
    for i, frame in enumerate(imageio.get_reader(str(path))):
        if i % every:
            continue
        frames.append(np.asarray(Image.fromarray(frame).resize(size, Image.BILINEAR)) if size else frame)
        if limit and len(frames) >= limit:
            break
    return np.stack(frames)


def pca_projector(features):
    """Fit three principal axes to token features (frames, tokens, dim) and return a function mapping any such features to RGB images (frames, GRID) on the same colour scale."""
    flat = features.reshape(-1, features.shape[-1]).astype(np.float32)
    mean = flat.mean(0)
    axes = np.linalg.eigh(np.cov((flat - mean).T))[1][:, -3:]
    low, high = np.percentile((flat - mean) @ axes, [2, 98], axis=0)
    return lambda x: np.clip(((x.astype(np.float32) - mean) @ axes - low) / (high - low), 0, 1).reshape(len(x), *GRID, 3)


class Window:
    """One six-frame window: RGB frames, true token images and the co-trained model's predicted token images for frames 1 to 5."""

    def __init__(self, rgb, features, model):
        batch = torch.from_numpy(features.astype(np.float32))[None]
        with torch.no_grad():
            predicted = run_model(model["encoder"], model["world_model"], batch, sample=False)[0][0].numpy()
        project = pca_projector(features)
        self.rgb, self.true, self.predicted = rgb, project(features), project(predicted)


class Assets:
    """Every clip, image and prediction the video needs, loaded once."""

    def __init__(self):
        self.human = [read_video(VIDEO_DIR / f"{name}.mp4", every=4, size=(384, 216), limit=120) for name in HUMAN_CLIPS]
        self.demos = [read_video(DEMO_DIR / "videos" / f"{name}.mp4") for name in DEMO_CLIPS]
        self.play = read_video(sorted((PLAY_DIR / "videos").glob("*.mp4"))[0], limit=160)
        self.rollouts = [read_video(ARTIFACT_DIR / "reach" / "task_eval" / c / t / f"seed_{s}" / "planner.mp4", every=2) for c, t, s, _ in ROLLOUTS]
        self.goal = imageio.imread(ARTIFACT_DIR / "reach" / "task_eval" / "cotrain_panda" / "reach" / "seed_3" / "goal.png")
        model = load_model(CHECKPOINT_DIR / "cotrain" / "last.pt", "cpu")
        name, start = HUMAN_WINDOW
        clip = read_video(VIDEO_DIR / f"{name}.mp4", size=(384, 224), limit=start + 5 * STRIDE + 1)
        idx = start + np.arange(6) * STRIDE
        self.human_window = Window(clip[idx], np.asarray(np.load(FEATURE_DIR / f"{name}.npy", mmap_mode="r")[idx]), model)
        name, start = PANDA_WINDOW
        idx = start + np.arange(6) * SIM_STRIDE
        with h5py.File(DEMO_DIR / "episodes" / f"{name}.h5") as f:
            rgb = f["rgb"][idx]
        self.panda_window = Window(rgb, np.load(DEMO_DIR / "features" / f"{name}.npy")[idx], model)


class Canvas:
    """A 16 x 9 drawing surface rendered to 1280 x 720 frames; every piece of text is set in Title Case."""

    def __init__(self):
        self.fig = plt.figure(figsize=(12.8, 7.2), dpi=100, facecolor=BG)
        self.ax = None

    def clear(self):
        """Start a new frame."""
        self.fig.clf()
        self.ax = self.fig.add_axes([0, 0, 1, 1])
        self.ax.set(xlim=(0, 16), ylim=(0, 9), aspect="equal")
        self.ax.axis("off")

    def render(self):
        """Return the current frame as an RGB array."""
        self.fig.canvas.draw()
        return np.asarray(self.fig.canvas.buffer_rgba())[..., :3].copy()

    def text(self, x, y, s, size=18, alpha=1.0, colour=INK, weight="normal", ha="center", va="center", raw=False):
        """Write text at data coordinates, in Title Case unless `raw` (links and maths)."""
        if alpha > 0:
            self.ax.text(x, y, s if raw or "$" in s else capitalise(s), fontsize=size, alpha=alpha, color=colour, fontweight=weight, ha=ha, va=va)

    def title(self, s, t, start=0.0):
        """Write the scene title."""
        self.text(8, 8.45, s, 25, ease(t, start), weight="bold")

    def image(self, img, x, y, w, alpha=1.0, label=None, label_colour=MUTED, label_size=12):
        """Draw an image with its left-bottom corner at (x, y) and width w, keeping its aspect ratio; return its height."""
        h = w * img.shape[0] / img.shape[1]
        if alpha > 0:
            self.ax.imshow(img, extent=(x, x + w, y, y + h), alpha=alpha, interpolation="bilinear")
        if label:
            self.text(x + w / 2, y - 0.25, label, label_size, alpha, label_colour)
        return h

    def outline(self, x, y, w, h, colour, alpha, lw=3.0):
        """Draw a highlight rectangle around a region."""
        if alpha > 0:
            self.ax.add_patch(Rectangle((x, y), w, h, fc="none", ec=colour, lw=lw, alpha=alpha))

    def box(self, x, y, w, h, colour, alpha=1.0, title=None, detail=None, size=15, title_dy=0.18, detail_dy=-0.22):
        """Draw an outlined rounded box centred on (x, y) with an optional title and detail line; return its anchors."""
        if alpha > 0:
            self.ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0,rounding_size=0.12", fc=colour, ec="none", alpha=alpha * 0.18))
            self.ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0,rounding_size=0.12", fc="none", ec=colour, lw=1.8, alpha=alpha))
        if title:
            self.text(x, y + (title_dy if detail else 0), title, size, alpha, INK, "bold")
        if detail:
            self.text(x, y + detail_dy, detail, size - 3, alpha, MUTED)
        return {"l": (x - w / 2, y), "r": (x + w / 2, y), "t": (x, y + h / 2), "b": (x, y - h / 2)}

    def arrow(self, p0, p1, alpha=1.0, colour=MUTED, dashed=False):
        """Draw an arrow between two points."""
        if alpha > 0:
            self.ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=15, lw=1.6, color=colour, alpha=alpha, ls="--" if dashed else "-", shrinkA=3, shrinkB=3))

    def line(self, p0, p1, alpha=1.0, colour=MUTED, dashed=False):
        """Draw a plain line between two points."""
        if alpha > 0:
            self.ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color=colour, lw=1.6, alpha=alpha, ls="--" if dashed else "-")


def clip_grid(c, clips, labels, t, start, step, cols=3, w=4.6, y0=4.85, x0=0.9, gap=0.4, speed=1.0, highlight=CLAY):
    """Play a grid of looping clips that appear one at a time, each announced by a coloured outline and label."""
    for i, clip in enumerate(clips):
        h = w * clip.shape[1] / clip.shape[2]
        x, y = x0 + (i % cols) * (w + gap), y0 - (i // cols) * (h + gap + 0.35)
        s = start + step * i
        c.image(clip[int(t * FPS * speed) % len(clip)], x, y, w, ease(t, s), labels[i], highlight, 14)
        c.outline(x - 0.06, y - 0.06, w + 0.12, h + 0.12, highlight, pulse(t, s, hold=step * 0.7))


def scene_title(c, a, t):
    """Opening card."""
    c.text(8, 5.0, "Latent Action World Model", 36, ease(t, 0.2, 0.8), weight="bold")
    c.text(8, 3.9, "github.com/VinayMohan0409/humanoid-submission", 15, ease(t, 1.2, 0.8), SLATE, raw=True)


def scene_human(c, a, t):
    """Egocentric data collection, one clip at a time."""
    c.title("Human Data: 64 Egocentric Clips", t)
    clip_grid(c, a.human, list(HUMAN_CLIPS.values()), t, 0.8, 1.4, speed=0.75)


def scene_robot(c, a, t):
    """Simulated data collection, one clip at a time, then what varies."""
    c.title("Robot Data: 30 Scripted Panda Demonstrations", t)
    clip_grid(c, a.demos, list(DEMO_CLIPS.values()), t, 0.6, 1.1, w=4.0, y0=5.45, x0=1.5)
    for i, line in enumerate(["Random Start And Goal Positions", "Random Objects And Clutter", "Some Grasps Fail"]):
        c.text(8, 1.2 - 0.45 * i, line, 15, ease(t, 7.6 + 0.9 * i), SAND if i == 2 else MUTED)


def frame_row(c, images, y, t, start, step=0.25, labels=None, alphas=None, outline=None):
    """Draw the six-column row of images used by the latent state and world model scenes."""
    for k, img in enumerate(images):
        alpha = ease(t, start + step * k) * (alphas[k] if alphas else 1)
        c.image(img, COL_X[k], y, COL_W, alpha, labels[k] if labels else None, INK, 15)
        if outline and outline[k] > 0:
            c.outline(COL_X[k] - 0.05, y - 0.05, COL_W + 0.1, COL_W * img.shape[0] / img.shape[1] + 0.1, SAGE, outline[k], 2.5)


def scene_state(c, a, t):
    """From frames to latent states."""
    c.title("Latent State", t)
    w = a.human_window
    frame_row(c, w.rgb, 5.7, t, 0.3, labels=[rf"$F_{k}$" for k in range(6)])
    c.arrow((8, 5.4), (8, 4.5), ease(t, 2.0), SLATE)
    c.text(8.3, 4.95, "Frozen DINOv3", 15, ease(t, 2.0), SLATE, ha="left")
    frame_row(c, w.true, 3.0, t, 2.5)
    c.text(8, 2.1, "DINOv3 Patch Tokens, Visualised With PCA", 16, ease(t, 4.3), MUTED)


def scene_world_model(c, a, t):
    """Pairs of latent states become latent actions through the encoder and bottleneck; then the causal decoder predicts each frame from F_0 and the earlier latent actions."""
    c.title("Latent Action World Model", t)
    w = a.human_window
    row_y = 3.0 + (6.5 - 3.0) * ease(t, 0.0, 1.2)
    decoding = t >= 10.0
    ghost = ease(t, 11.0, 0.8)
    steps = [ease(t, 12.5 + 1.6 * (k - 1), 0.5) for k in range(1, 6)]  # predicted frame k fades in over the true one
    images = [w.true[0]] + [w.true[k] * (1 - steps[k - 1]) + w.predicted[k - 1] * steps[k - 1] for k in range(1, 6)]
    labels = [r"$F_0$"] + [rf"$\hat{{F}}_{k}$" if steps[k - 1] > 0.5 else rf"$F_{k}$" for k in range(1, 6)]
    alphas = [1.0] + [max(1 - 0.75 * ghost, steps[k - 1]) for k in range(1, 6)]
    frame_row(c, images, row_y, t, -1.0, 0.0, labels, alphas, [0.0] + steps)
    h = COL_W * GRID[0] / GRID[1]
    encode = 1 - ease(t, 9.5, 0.8)
    for p in range(5):  # one latent action per pair of frames
        glow = pulse(t, 1.5 + 1.6 * p, hold=1.0) * encode
        c.outline(COL_X[p] - 0.08, row_y - 0.08, 2.5 + COL_W + 0.16, h + 0.16, CLAY, glow)
        c.arrow((COL_X[p] + COL_W + 0.15, row_y - 0.1), (8, 5.3), glow, CLAY)
    enc = c.box(8, 4.6, 4.8, 1.35, SLATE, ease(t, 1.5) * encode, "Latent Action Encoder", "Self-Attention Over Two Frames", 16)
    bott = c.box(8, 2.9, 4.6, 0.8, SLATE, ease(t, 1.9) * encode, "Bottleneck: 2 Tokens × 8 Numbers", size=14)
    c.arrow(enc["b"], bott["t"], ease(t, 1.9) * encode, SLATE)
    dec = c.box(8, 4.3, 5.6, 1.5, SAGE, ease(t, 10.3), "Causal Latent Action Decoder", "(World Model)", 17)
    chip_y, chip_w = 1.45, 1.35
    lit = [max(steps[k:]) for k in range(5)] if decoding else [1.0] * 5  # while decoding, chip k is lit once a predicted frame that uses it has appeared
    for k in range(5):
        x = 3.6 + k * 2.2
        appear = ease(t, 2.6 + 1.6 * k)
        c.box(x, chip_y, chip_w, 0.7, SAND, appear * (0.3 + 0.7 * lit[k]), rf"$z_{k + 1}$", size=16)
        c.arrow((x, chip_y + 0.35), (8 + (x - 8) * 0.35, dec["b"][1]), ease(t, 10.3) * lit[k] * 0.8, SAND)
    c.arrow(bott["b"], (8, chip_y + 0.4), ease(t, 2.6) * encode, SAND)
    c.text(8, 0.5, "One Latent Action For Each Pair Of Frames", 15, ease(t, 3.2) * encode, MUTED)
    c.arrow((COL_X[0] + COL_W / 2, row_y - 0.1), (5.3, dec["t"][1]), ease(t, 11.0) * 0.8, STONE)
    c.text(2.3, 5.35, r"Only $F_0$ Goes In", 13, ease(t, 11.0), MUTED)
    c.text(8, 0.5, r"Causal: Predicted Frame $\hat{F}_k$ Uses Only $z_1 \dots z_k$", 15, ease(t, 13.5), MUTED)


def scene_fidelity(c, a, t):
    """Held-out predictions: RGB, true tokens and predicted tokens appear column by column, human then Panda."""
    c.title("Prediction On Held-Out Clips", t)
    half = 5.5
    w, base = (a.human_window, 0.4) if t < half else (a.panda_window, half + 0.3)
    for row, (images, label) in enumerate(((w.rgb, "RGB"), (w.true, "True DINOv3"), (w.predicted, "Predicted DINOv3"))):
        y = 5.85 - 1.95 * row
        c.text(0.55, y + 0.62, label, 13, ease(t, base), MUTED, ha="left")
        for k, img in enumerate(images):
            col = k + (1 if row == 2 else 0)
            x = 2.6 + col * 2.25
            c.image(img, x, y, 2.0, ease(t, base + 0.45 * col))
            if row == 0:
                c.text(x + 1.0, y + 1.45, rf"$F_{col}$", 14, ease(t, base + 0.45 * col))
    c.text(8, 0.55, "Human Clip" if t < half else "Panda Demonstration", 14, ease(t, base), MUTED)


def scene_idm(c, a, t):
    """The inverse dynamics model, trained separately on labelled play data."""
    c.title("Inverse Dynamics Model", t)
    c.image(a.play[int(t * FPS) % len(a.play)], 0.9, 3.5, 6.2, ease(t, 0.3), "Play Data With Action Labels", INK, 15)
    dino = c.box(9.7, 5.3, 2.4, 1.1, SLATE, ease(t, 2.5), "Frozen DINOv3", size=15)
    idm = c.box(13.4, 5.3, 3.8, 1.1, SAGE, ease(t, 4.0), "Inverse Dynamics Model", size=15)
    c.arrow((7.2, 5.3), dino["l"], ease(t, 2.5), SLATE)
    c.arrow(dino["r"], idm["l"], ease(t, 4.0), SAGE)
    c.text(13.4, 4.3, "Two Frames → Robot Action", 13, ease(t, 4.0), MUTED)
    c.text(8, 2.3, "Turns Latent Actions Into Robot Actions", 19, ease(t, 5.5), INK, "bold")
    c.text(8, 1.6, "The World Model Never Sees These Labels", 16, ease(t, 7.0), MUTED)


def scene_planning(c, a, t):
    """Closed-loop goal-image planning."""
    c.title("Planning: Goal Image In, Commands Out", t)
    c.image(a.rollouts[0][0], 0.7, 5.3, 3.3, ease(t, 0.3), "Current Frame", INK, 13)
    c.image(a.goal, 0.7, 2.3, 3.3, ease(t, 0.9), "Goal Image", INK, 13)
    dino = c.box(5.9, 4.6, 2.0, 1.1, SLATE, ease(t, 2.0), "DINOv3", size=15)
    cem = c.box(9.1, 4.6, 3.5, 1.1, CLAY, ease(t, 3.0), "CEM Over Latent Actions", size=15)
    wm = c.box(12.3, 4.6, 2.3, 1.1, SAGE, ease(t, 4.0), "World Model", size=15)
    idm = c.box(14.8, 4.6, 1.8, 1.1, SAGE, ease(t, 5.0), "IDM", size=15)
    c.arrow((4.1, 6.2), dino["l"], ease(t, 2.0), SLATE)
    c.arrow((4.1, 3.2), dino["l"], ease(t, 2.0), SLATE)
    for p, q, s in ((dino, cem, 3.0), (cem, wm, 4.0), (wm, idm, 5.0)):
        c.arrow(p["r"], q["l"], ease(t, s), SAGE)
    c.arrow((14.8, 4.05), (14.8, 2.85), ease(t, 6.0), MUTED)
    c.text(14.8, 2.55, "Simulator", 14, ease(t, 6.0), MUTED)
    c.line((14.8, 2.3), (14.8, 1.5), ease(t, 7.0), MUTED, True)
    c.arrow((14.8, 1.5), (2.35, 1.5), ease(t, 7.0), MUTED, True)
    c.text(8, 1.85, "Closed Loop: Replan Every 0.2 Seconds", 15, ease(t, 7.0), SAND)


def scene_rollouts(c, a, t):
    """One rollout at a time, large."""
    c.title("Planning In Simulation", t)
    each = 5.0
    i = min(int(t // each), len(ROLLOUTS) - 1)
    local = t - i * each
    clip = a.rollouts[i]
    frame = clip[min(int(local * FPS * 1.5), len(clip) - 1)]
    c.image(frame, 3.2, 2.1, 9.6, ease(local, 0.0, 0.4), ROLLOUTS[i][3], INK, 16)
    c.text(8, 0.75, "Green: Goal Gripper Pose · 3× Speed", 13, ease(t, 0.5), MUTED)


def scene_table(c, a, t):
    """The success counts."""
    c.title("Success: End Effector Within 5 Centimetres Of The Goal", t)
    c.text(8.6, 6.6, "Reach (5 Tasks)", 17, ease(t, 0.2), MUTED)
    c.text(12.3, 6.6, "Carry (10 Tasks)", 17, ease(t, 0.2), MUTED)
    rows = [("Co-Trained + Panda Gripper", "5 / 5", "6 / 10", SAGE), ("Co-Trained + Robotiq Gripper", "3 / 5", "5 / 10", SAGE), ("Human-Only + Panda Gripper", "1 / 5", "2 / 10", CLAY)]
    for i, (name, reach, carry, colour) in enumerate(rows):
        alpha = ease(t, 0.8 + 0.9 * i)
        c.text(1.6, 5.5 - 1.0 * i, name, 18, alpha, colour, "bold", ha="left")
        c.text(8.6, 5.5 - 1.0 * i, reach, 20, alpha, INK)
        c.text(12.3, 5.5 - 1.0 * i, carry, 20, alpha, INK)


def scene_end(c, a, t):
    """Closing card."""
    c.text(8, 5.0, "Thank You For Watching", 30, ease(t, 0.2, 0.8), weight="bold")
    c.text(8, 3.9, "github.com/VinayMohan0409/humanoid-submission", 15, ease(t, 1.0, 0.8), SLATE, raw=True)


SCENES = [(3.5, scene_title), (10.5, scene_human), (11.5, scene_robot), (7.0, scene_state), (21.5, scene_world_model), (11.0, scene_fidelity), (9.0, scene_idm), (9.0, scene_planning), (15.0, scene_rollouts), (5.0, scene_table), (3.5, scene_end)]
PREVIEW_TIMES = {"scene_world_model": (3.0, 8.0, 14.0, 20.0), "scene_fidelity": (4.5, 10.0), "scene_human": (3.0, 9.5), "scene_robot": (4.0, 10.5)}


def main():
    """Render every scene in order to one mp4, or preview frames with --preview."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--preview", action="store_true", help="write preview PNGs instead of the video")
    args = parser.parse_args()
    assets, canvas = Assets(), Canvas()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.preview:
        for i, (duration, scene) in enumerate(SCENES):
            for j, t in enumerate(PREVIEW_TIMES.get(scene.__name__, (duration * 0.85,))):
                canvas.clear()
                scene(canvas, assets, t)
                imageio.imwrite(OUT / f"preview_{i:02d}_{j}_{scene.__name__}.png", canvas.render())
        return
    writer = imageio.get_writer(str(OUT / "explainer.mp4"), fps=FPS, codec="libx264", quality=8, pixelformat="yuv420p", macro_block_size=1)
    background = np.array(matplotlib.colors.to_rgb(BG)) * 255
    for duration, scene in SCENES:
        for frame in range(int(duration * FPS)):
            t = frame / FPS
            canvas.clear()
            scene(canvas, assets, t)
            fade = min(1.0, t / 0.3, (duration - t) / 0.3)  # a short fade between scenes
            writer.append_data((canvas.render() * fade + background * (1 - fade)).astype(np.uint8))
        print(scene.__name__, flush=True)
    writer.close()
    print(OUT / "explainer.mp4")


if __name__ == "__main__":
    main()
