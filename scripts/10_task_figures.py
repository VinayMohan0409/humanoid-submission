"""Summarise the task evaluation: write the results table (docs/task_results.md) and one grid video per task type, with a column per condition and a row per seed, the green silhouette marking the goal pose."""
import json

import imageio.v3 as iio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from src.config import ARTIFACT_DIR, FIGURE_DIR

RESULTS = ARTIFACT_DIR / "reach" / "task_eval"
DOCS = FIGURE_DIR.parent
VIDEO_DIR = DOCS / "videos"
CONDITIONS = {"human_only_panda": "Human-Only, Panda Gripper", "cotrain_panda": "Co-Trained, Panda Gripper", "cotrain_robotiq": "Co-Trained, Robotiq Gripper"}
TASKS = {"reach": "Reach", "carry": "Carry"}
CELL = (288, 168)  # width and height of one rollout in the grid
SHAPES = ("cube", "cuboid", "cylinder")  # carried object of seed k is SHAPES[k % 3]


def load(condition):
    """Return the result rows of one condition."""
    return json.load(open(RESULTS / condition / "results.json"))


def table(results):
    """Return the Markdown results table: successes out of trials and the mean start and end distance to the goal per condition and task."""
    lines = ["| Condition | Task | Success | Mean Start Distance | Mean End Distance |", "|---|---|---|---|---|"]
    for condition, label in CONDITIONS.items():
        for task, task_label in TASKS.items():
            rows = [r for r in results[condition] if r["task"] == task]
            mean = lambda key: sum(r[key] for r in rows) / len(rows) * 100
            lines.append(f"| {label} | {task_label} | {sum(r['success'] for r in rows)} / {len(rows)} | {mean('start_m'):.1f} cm | {mean('end_m'):.1f} cm |")
    return "\n".join(lines) + "\n"


def read_clip(condition, task, seed, size):
    """Return a rollout video's frames resized to `size` (width, height), the goal silhouette included."""
    return [np.array(Image.fromarray(frame).resize(size, Image.BILINEAR)) for frame in iio.imread(RESULTS / condition / task / f"seed_{seed}" / "planner.mp4")]


def label_frame(frame, text, font):
    """Draw `text` on the bottom-left of a frame, white on a dark strip so it reads on any background."""
    image = Image.fromarray(frame)
    draw = ImageDraw.Draw(image)
    draw.rectangle([0, image.height - 16, image.width, image.height], fill=(20, 20, 20))
    draw.text((4, image.height - 15), text, fill=(255, 255, 255), font=font)
    return np.array(image)


def save_videos(path, results, task):
    """Write one grid video of every seed of one task type, a column per condition and a row per seed, each cell labelled with its seed and outcome; rollouts that stopped early hold their last frame. Saves an mp4 and a smaller looping GIF for the README."""
    seeds = sorted({r["seed"] for r in results["cotrain_panda"] if r["task"] == task})
    font = ImageFont.load_default(size=12)
    rows = []
    for seed in seeds:
        cells = []
        for condition in CONDITIONS:
            r = next(r for r in results[condition] if r["task"] == task and r["seed"] == seed)
            name = f"seed {seed}" + (f" {SHAPES[seed % 3]}" if task == "carry" else "")
            clip = [label_frame(f, f"{name}: {r['end_m'] * 100:.1f} cm, {'reached' if r['success'] else 'missed'}", font) for f in read_clip(condition, task, seed, CELL)]
            cells.append(clip)
        rows.append(cells)
    length = max(len(clip) for cells in rows for clip in cells)
    pad = lambda clip: clip + [clip[-1]] * (length - len(clip))
    header = np.full((26, CELL[0] * len(CONDITIONS), 3), 252, dtype=np.uint8)
    head = Image.fromarray(header)
    for col, label in enumerate(CONDITIONS.values()):
        ImageDraw.Draw(head).text((col * CELL[0] + 6, 5), label, fill=(11, 11, 11), font=ImageFont.load_default(size=14))
    frames = [np.concatenate([np.array(head)] + [np.concatenate([pad(clip)[t] for clip in cells], axis=1) for cells in rows], axis=0) for t in range(length)]
    iio.imwrite(path.with_suffix(".mp4"), frames, fps=20)
    small = [Image.fromarray(f).resize((f.shape[1] // 3 * 2, f.shape[0] // 3 * 2), Image.BILINEAR).quantize(64, dither=Image.Dither.NONE) for f in frames[::4]]
    small[0].save(path.with_suffix(".gif"), save_all=True, append_images=small[1:], duration=200, loop=0, optimize=True)


def main():
    """Write the table and the two grids."""
    results = {condition: load(condition) for condition in CONDITIONS}
    (DOCS / "task_results.md").write_text(table(results))
    VIDEO_DIR.mkdir(parents=True, exist_ok=True)
    for task in TASKS:
        save_videos(VIDEO_DIR / f"{task}_examples", results, task)
    print(table(results))


if __name__ == "__main__":
    main()
