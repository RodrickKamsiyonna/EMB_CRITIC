"""Full-episode HDF5 dataset with frameskip subsampling and dynamic batch padding.

Loads entire episodes (at `frameskip`, e.g. 5) instead of stride-1 sliding windows:
- Every frame is encoded by ResNet-18 once per epoch (~20-25x faster per epoch).
- The EBM Transformer trains directly on full episode sequences (20-64+ frames),
  so 25-step (5-frame), 50-step (10-frame), and 50-frame horizons are in-distribution.
- Variable-length episodes are dynamically padded per batch with a boolean `valid_mask`,
  and padded frames are excluded from ResNet-18, SIGReg, Energy sum, and MSE loss.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import h5py
import hdf5plugin  # noqa: F401
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms import v2 as T


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class EpisodeHDF5Dataset(Dataset):
    """Loads full episodes from a stable-worldmodel HDF5 file at `frameskip`."""

    def __init__(
        self,
        h5_path: str | Path,
        frameskip: int = 5,
        img_size: int = 224,
        max_ep_frames: int = 64,
        random_crop: bool = True,
        action_mean: Optional[torch.Tensor] = None,
        action_std: Optional[torch.Tensor] = None,
    ):
        self.h5_path = Path(h5_path)
        if not self.h5_path.exists():
            raise FileNotFoundError(f"HDF5 dataset not found: {self.h5_path}")

        self.frameskip = int(frameskip)
        self.img_size = int(img_size)
        self.max_ep_frames = int(max_ep_frames)
        self.random_crop = bool(random_crop)
        self._h5_file: Optional[h5py.File] = None

        with h5py.File(self.h5_path, "r", swmr=True) as f:
            self.ep_len = f["ep_len"][:].astype(np.int64)
            self.ep_offset = f["ep_offset"][:].astype(np.int64)
            actions = f["action"][:]

        # Filter out degenerate episodes that don't have at least 2 frames at `frameskip`
        min_raw_len = self.frameskip + 1
        self.valid_ep_indices = np.nonzero(self.ep_len >= min_raw_len)[0]

        # Cache actions in RAM (small) and compute z-score stats matching LeWM
        self._actions = actions.astype(np.float32, copy=False)
        self.raw_action_dim = int(np.prod(self._actions.shape[1:])) if self._actions.ndim > 1 else 1
        self.action_dim = self.frameskip * self.raw_action_dim

        if action_mean is not None and action_std is not None:
            self.action_mean = action_mean.float().reshape(1, self.raw_action_dim)
            self.action_std = action_std.float().reshape(1, self.raw_action_dim)
        else:
            flat_act = self._actions.reshape(-1, self.raw_action_dim)
            valid_rows = ~np.isnan(flat_act).any(axis=1)
            valid_act = torch.from_numpy(flat_act[valid_rows])
            self.action_mean = valid_act.mean(dim=0, keepdim=True).float()
            self.action_std = valid_act.std(dim=0, keepdim=True).clamp(min=1e-6).float()

        self.resize = T.Resize((self.img_size, self.img_size), antialias=True)
        self.img_mean = torch.tensor(IMAGENET_MEAN, dtype=torch.float32).view(1, 3, 1, 1)
        self.img_std = torch.tensor(IMAGENET_STD, dtype=torch.float32).view(1, 3, 1, 1)

    def _get_h5(self) -> h5py.File:
        if self._h5_file is None:
            self._h5_file = h5py.File(
                self.h5_path, "r", swmr=True, rdcc_nbytes=256 * 1024 * 1024
            )
        return self._h5_file

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_h5_file"] = None
        return state

    def __len__(self) -> int:
        return len(self.valid_ep_indices)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        ep_id = int(self.valid_ep_indices[idx])
        offset = int(self.ep_offset[ep_id])
        raw_len = int(self.ep_len[ep_id])
        fs = self.frameskip

        # Number of sampled frames at stride `frameskip`: indices 0, fs, 2*fs, ..., (n_frames-1)*fs < raw_len
        n_frames = (raw_len - 1) // fs + 1
        start_frame = 0
        if n_frames > self.max_ep_frames:
            if self.random_crop:
                start_frame = int(torch.randint(0, n_frames - self.max_ep_frames + 1, (1,)).item())
            n_frames = self.max_ep_frames

        raw_start = offset + start_frame * fs
        raw_end = raw_start + (n_frames - 1) * fs + 1

        # Read only the subsampled frames via HDF5 strided slice
        f = self._get_h5()
        raw_pixels = f["pixels"][raw_start:raw_end:fs]  # (T, H, W, 3) or (T, 3, H, W)
        pixels = torch.from_numpy(np.ascontiguousarray(raw_pixels))
        if pixels.ndim == 4 and pixels.shape[-1] in (1, 3):
            pixels = pixels.permute(0, 3, 1, 2)  # (T, C, H, W)

        if pixels.dtype == torch.uint8:
            pixels = pixels.float().div_(255.0)
        else:
            pixels = pixels.float()
            if pixels.max() > 1.5:
                pixels = pixels.div_(255.0)

        if pixels.shape[-2] != self.img_size or pixels.shape[-1] != self.img_size:
            pixels = self.resize(pixels)
        pixels = (pixels - self.img_mean) / self.img_std

        # Build frameskip-concatenated normalized action blocks for each transition k -> k+1.
        # Last frame (k = n_frames - 1) has no outgoing transition, so fill with NaN (masked in IDM).
        n_trans = n_frames - 1
        act_slice = self._actions[raw_start : raw_start + n_trans * fs].reshape(
            n_trans * fs, self.raw_action_dim
        )
        act_t = (torch.from_numpy(act_slice.copy()) - self.action_mean) / self.action_std
        act_blocks = act_t.reshape(n_trans, self.action_dim)  # (T-1, fs * raw_act_dim)

        pad_last = torch.full((1, self.action_dim), float("nan"), dtype=torch.float32)
        actions = torch.cat([act_blocks, pad_last], dim=0)  # (T, action_dim)

        return {
            "pixels": pixels,
            "action": actions,
            "length": torch.tensor(n_frames, dtype=torch.long),
        }


def pad_episode_collate(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Dynamically pad variable-length episodes to T_max in batch and create boolean valid_mask."""
    b = len(batch)
    lengths = torch.stack([item["length"] for item in batch])  # (B,)
    t_max = int(lengths.max().item())

    c, h, w = batch[0]["pixels"].shape[1:]
    act_dim = batch[0]["action"].shape[-1]

    pixels = torch.zeros((b, t_max, c, h, w), dtype=batch[0]["pixels"].dtype)
    actions = torch.full((b, t_max, act_dim), float("nan"), dtype=torch.float32)
    valid_mask = torch.zeros((b, t_max), dtype=torch.bool)

    for i, item in enumerate(batch):
        t = int(item["length"].item())
        pixels[i, :t] = item["pixels"]
        actions[i, :t] = item["action"]
        valid_mask[i, :t] = True

    return {
        "pixels": pixels,
        "action": actions,
        "valid_mask": valid_mask,
        "lengths": lengths,
    }
