"""A device-aware ring buffer for conditional Drifting training."""

from __future__ import annotations

from collections.abc import Sequence

import torch


class TensorMemoryBank:
    """Store sample tensors in per-bucket ring buffers.

    The storage stays on CPU by default so large negative banks do not consume
    scarce GPU memory. ``sample`` optionally moves the selected subset to the
    requested device.
    """

    def __init__(
        self,
        num_buckets: int,
        max_size: int,
        feature_shape: Sequence[int] | None = None,
        dtype: torch.dtype = torch.float32,
        device: str | torch.device = "cpu",
    ) -> None:
        if num_buckets <= 0 or max_size <= 0:
            raise ValueError("num_buckets and max_size must be positive")
        self.num_buckets = int(num_buckets)
        self.max_size = int(max_size)
        self.feature_shape = None if feature_shape is None else tuple(int(v) for v in feature_shape)
        self.dtype = dtype
        self.device = torch.device(device)
        self.bank: torch.Tensor | None = None
        self.ptr = torch.zeros(self.num_buckets, dtype=torch.long)
        self.count = torch.zeros(self.num_buckets, dtype=torch.long)

    def _initialize(self, feature_shape: Sequence[int]) -> None:
        resolved = tuple(int(v) for v in feature_shape)
        if self.feature_shape is not None and self.feature_shape != resolved:
            raise ValueError(f"expected feature shape {self.feature_shape}, got {resolved}")
        self.feature_shape = resolved
        self.bank = torch.zeros(
            (self.num_buckets, self.max_size, *resolved),
            dtype=self.dtype,
            device=self.device,
        )

    def add(self, samples: torch.Tensor, buckets: torch.Tensor) -> None:
        if samples.ndim < 2:
            raise ValueError("samples must include a leading sample dimension")
        if samples.shape[0] != buckets.numel():
            raise ValueError("samples and buckets must contain the same number of items")
        if self.bank is None:
            self._initialize(samples.shape[1:])
        assert self.bank is not None
        if tuple(samples.shape[1:]) != self.feature_shape:
            raise ValueError(f"expected feature shape {self.feature_shape}, got {tuple(samples.shape[1:])}")

        bucket_ids = buckets.detach().to(device="cpu", dtype=torch.long).reshape(-1)
        if bucket_ids.numel() and (bucket_ids.min() < 0 or bucket_ids.max() >= self.num_buckets):
            raise ValueError("bucket index out of range")
        values = samples.detach().to(device=self.device, dtype=self.dtype)
        for value, bucket in zip(values, bucket_ids.tolist()):
            index = int(self.ptr[bucket].item())
            self.bank[bucket, index].copy_(value)
            self.ptr[bucket] = (index + 1) % self.max_size
            self.count[bucket] = min(int(self.count[bucket].item()) + 1, self.max_size)

    def sample(
        self,
        buckets: torch.Tensor,
        n_samples: int,
        *,
        output_device: str | torch.device | None = None,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        if n_samples <= 0:
            raise ValueError("n_samples must be positive")
        if self.bank is None or self.feature_shape is None:
            raise RuntimeError("memory bank is empty; call add() first")

        bucket_ids = buckets.detach().to(device="cpu", dtype=torch.long).reshape(-1)
        if bucket_ids.numel() and (bucket_ids.min() < 0 or bucket_ids.max() >= self.num_buckets):
            raise ValueError("bucket index out of range")
        indices = torch.zeros((bucket_ids.numel(), n_samples), dtype=torch.long)
        for row, bucket in enumerate(bucket_ids.tolist()):
            valid = int(self.count[bucket].item())
            if valid == 0:
                continue
            if valid >= n_samples:
                indices[row] = torch.randperm(valid, generator=generator)[:n_samples]
            else:
                indices[row] = torch.randint(
                    valid,
                    (n_samples,),
                    generator=generator,
                    dtype=torch.long,
                )

        sampled = self.bank[bucket_ids[:, None], indices]
        if output_device is not None:
            sampled = sampled.to(output_device)
        return sampled

    def available(self, bucket: int) -> int:
        return int(self.count[int(bucket)].item())

    def clear(self) -> None:
        if self.bank is not None:
            self.bank.zero_()
        self.ptr.zero_()
        self.count.zero_()
