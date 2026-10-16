"""Simulated demonstration windows: DINO features of frames SIM_STRIDE control steps apart, with the robot actions and gripper state between them."""
import h5py
import numpy as np
import torch

from src.config import CHUNK, DEMO_DIR, FEATURE_DIM, PLAY_DIR, SIM_STRIDE
from src.datasets.human import pool

FINGER_OPEN = 0.04  # finger joint position when fully open, used to scale the gripper state to about 1


DEMOS, PLAY = (DEMO_DIR, 5), (PLAY_DIR, 10)  # (folder, hold out every n-th episode as "val")


def sim_split(split, folder, every):
    """Return the episode names of a folder: every `every`-th episode is held out as "val"; "train" is the rest and "all" is every episode."""
    names = sorted(p.stem for p in (folder / "episodes").glob("ep_*.h5"))
    return [name for i, name in enumerate(names) if split == "all" or (i % every == every - 1) == (split == "val")]


class SimChunkDataset:
    """Every window of CHUNK + 1 frames SIM_STRIDE apart from the first `max_episodes` (default all) episodes of one split of each source, with its actions and per-pair gripper state, resident on `device`."""

    def __init__(self, split, device, sources=(DEMOS,), max_episodes=None):
        frames, actions, grips, f_starts, a_starts = [], [], [], [], []
        f_off = a_off = 0
        for folder, name in [(folder, name) for folder, every in sources for name in sim_split(split, folder, every)[:max_episodes]]:
            with h5py.File(folder / "episodes" / f"{name}.h5") as f:
                action, grip = f["action"][:], f["gripper_qpos"][:]
            feats = np.load(folder / "features" / f"{name}.npy")
            starts = np.arange(len(action) - CHUNK * SIM_STRIDE + 1)
            f_starts.append(f_off + starts)
            a_starts.append(a_off + starts)
            frames.append(torch.from_numpy(feats))
            actions.append(torch.from_numpy(action))
            grips.append(torch.from_numpy(grip).float())
            f_off, a_off = f_off + len(feats), a_off + len(action)
        to = lambda x: x.to(device)
        self.frames, self.actions, self.grips = to(torch.cat(frames)), to(torch.cat(actions)), to(torch.cat(grips))
        self.f_starts, self.a_starts = to(torch.from_numpy(np.concatenate(f_starts))), to(torch.from_numpy(np.concatenate(a_starts)))
        self.f_offsets = to(torch.arange(CHUNK + 1) * SIM_STRIDE)
        self.a_offsets = to(torch.arange(CHUNK)[:, None] * SIM_STRIDE + torch.arange(SIM_STRIDE)[None, :])

    def __len__(self):
        """Return the number of windows in the split."""
        return len(self.f_starts)

    def batches(self, batch_size, shuffle, generator=None):
        """Yield (frames (B, CHUNK + 1, tokens, 384), actions (B, CHUNK, SIM_STRIDE, 7), gripper state (B, CHUNK, 2))."""
        order = torch.randperm(len(self), generator=generator) if shuffle else torch.arange(len(self))
        for i in range(0, len(self), batch_size):
            idx = order[i : i + batch_size].to(self.f_starts.device)
            f, a = self.f_starts[idx], self.a_starts[idx]
            window = self.frames[f[:, None] + self.f_offsets].float()
            frames = pool(window.flatten(0, 1)).view(len(idx), CHUNK + 1, -1, FEATURE_DIM)
            yield frames, self.actions[a[:, None, None] + self.a_offsets], self.grips[f[:, None] + self.f_offsets[:-1]] / FINGER_OPEN


class SimFrames:
    """The simulated demo windows as bare frame tensors, so they go through the same loss and evaluation as the human clips."""

    def __init__(self, split, device):
        self.data = SimChunkDataset(split, device, (DEMOS,))

    def __len__(self):
        """Return the number of windows."""
        return len(self.data)

    def batches(self, batch_size, shuffle, generator=None, drop_last=False):
        """Yield (B, CHUNK + 1, tokens, 384) frame windows, dropping the actions."""
        for frames, *_ in self.data.batches(batch_size, shuffle, generator):
            yield frames
