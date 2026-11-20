"""Deterministic, DDP-aware sampler supporting exact mid-epoch dataset resume."""

from __future__ import annotations

import math
import os
from typing import Iterator, Optional, Tuple

import torch
import torch.distributed as dist
from torch.utils.data import Sampler


class ResumableDistributedSampler(Sampler[int]):
    """Deterministic sampler that supports exact mid-epoch resume in single-GPU and DDP.

    Given a fixed `seed` and `epoch`, the dataset permutation is identical across runs and
    across all DDP ranks. When resuming mid-epoch from `start_batch`, this sampler skips the
    first `start_batch * batch_size` indices already processed on the current rank, then
    automatically resets so subsequent epochs iterate from batch 0.
    """

    def __init__(
        self,
        dataset_len: int,
        batch_size: int,
        seed: int = 0,
        shuffle: bool = True,
        drop_last: bool = True,
        rank: Optional[int] = None,
        world_size: Optional[int] = None,
    ):
        if dataset_len <= 0:
            raise ValueError(f"dataset_len must be > 0, got {dataset_len}")
        if batch_size <= 0:
            raise ValueError(f"batch_size must be > 0, got {batch_size}")

        self.dataset_len = int(dataset_len)
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.shuffle = bool(shuffle)
        self.drop_last = bool(drop_last)
        self._explicit_rank = rank
        self._explicit_world_size = world_size

        self.epoch: int = 0
        self.resume_epoch: Optional[int] = None
        self.resume_start_batch: int = 0

    def _get_dist_info(self) -> Tuple[int, int]:
        if self._explicit_rank is not None and self._explicit_world_size is not None:
            return int(self._explicit_rank), int(self._explicit_world_size)
        if dist.is_available() and dist.is_initialized():
            return dist.get_rank(), dist.get_world_size()
        rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", 0)))
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        return rank, max(1, world_size)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def set_resume(self, epoch: Optional[int], start_batch: int) -> None:
        self.resume_epoch = int(epoch) if epoch is not None else None
        self.resume_start_batch = max(0, int(start_batch))

    def _num_samples_per_rank(self, world_size: int) -> int:
        if self.drop_last:
            # Ensure every rank gets an exact multiple of batch_size
            global_batch = world_size * self.batch_size
            num_global_batches = self.dataset_len // global_batch
            return num_global_batches * self.batch_size
        return math.ceil(self.dataset_len / world_size)

    def num_batches_per_rank(self) -> int:
        _, world_size = self._get_dist_info()
        n = self._num_samples_per_rank(world_size)
        return n // self.batch_size if self.drop_last else math.ceil(n / self.batch_size)

    def _build_rank_indices(self, epoch: int, rank: int, world_size: int):
        num_per_rank = self._num_samples_per_rank(world_size)
        total_size = num_per_rank * world_size

        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed + int(epoch))
            indices = torch.randperm(self.dataset_len, generator=g).tolist()
        else:
            indices = list(range(self.dataset_len))

        if total_size > len(indices):
            padding_size = total_size - len(indices)
            if padding_size <= len(indices):
                indices += indices[:padding_size]
            else:
                indices += (indices * math.ceil(padding_size / len(indices)))[:padding_size]
        else:
            indices = indices[:total_size]

        # Interleaved sharding across ranks (matches torch DistributedSampler)
        return indices[rank:total_size:world_size]

    def __iter__(self) -> Iterator[int]:
        rank, world_size = self._get_dist_info()
        rank_indices = self._build_rank_indices(self.epoch, rank, world_size)

        if (
            self.resume_epoch is not None
            and self.resume_epoch == self.epoch
            and self.resume_start_batch > 0
        ):
            offset = min(len(rank_indices), self.resume_start_batch * self.batch_size)
            rank_indices = rank_indices[offset:]
            # Consume resume state once so the next epoch iterates from batch 0
            self.resume_epoch = None
            self.resume_start_batch = 0

        return iter(rank_indices)

    def __len__(self) -> int:
        _, world_size = self._get_dist_info()
        n = self._num_samples_per_rank(world_size)
        if (
            self.resume_epoch is not None
            and self.resume_epoch == self.epoch
            and self.resume_start_batch > 0
        ):
            return max(0, n - self.resume_start_batch * self.batch_size)
        return n
