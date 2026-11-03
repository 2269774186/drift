"""Training step for the YODA-conditioned Drifting generator."""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F

from .drift_loss import latent_drift_loss
from .drift_model import ConditionalDriftGenerator
from .memory_bank import TensorMemoryBank


class ConditionalDriftTrainer:
    """Train a one-forward generator with positive and historical negatives."""

    def __init__(
        self,
        generator: ConditionalDriftGenerator,
        latent_shape: Sequence[int],
        *,
        num_buckets: int = 1,
        max_positive: int = 32,
        max_negative: int = 32,
        positive_weight: float = 1.0,
        negative_weight: float = 1.0,
        reconstruction_weight: float = 0.05,
        R_list: Sequence[float] = (0.2, 0.05, 0.02),
        bank_device: str | torch.device = "cpu",
    ) -> None:
        if positive_weight < 0 or negative_weight < 0 or reconstruction_weight < 0:
            raise ValueError("loss weights must be non-negative")
        if num_buckets <= 0:
            raise ValueError("num_buckets must be positive")
        self.generator = generator
        self.positive_bank = TensorMemoryBank(
            num_buckets,
            max_positive,
            feature_shape=latent_shape,
            device=bank_device,
        )
        self.negative_bank = TensorMemoryBank(
            num_buckets,
            max_negative,
            feature_shape=latent_shape,
            device=bank_device,
        )
        self.positive_weight = float(positive_weight)
        self.negative_weight = float(negative_weight)
        self.reconstruction_weight = float(reconstruction_weight)
        self.R_list = tuple(R_list)

    def step(
        self,
        lq_latent_hat: torch.Tensor,
        target_latent: torch.Tensor,
        ctxs: Sequence[torch.Tensor] | None,
        optimizer: torch.optim.Optimizer,
        *,
        noise: torch.Tensor | None = None,
        bucket_ids: torch.Tensor | None = None,
        max_grad_norm: float = 2.0,
    ) -> dict[str, float]:
        if lq_latent_hat.shape != target_latent.shape:
            raise ValueError("condition and target latents must have the same shape")
        if lq_latent_hat.ndim != 4:
            raise ValueError("lq_latent_hat and target_latent must have shape (B, C, H, W)")

        batch_size = lq_latent_hat.shape[0]
        if bucket_ids is None:
            buckets = torch.zeros(batch_size, dtype=torch.long)
        else:
            buckets = bucket_ids.detach().to(device="cpu", dtype=torch.long).reshape(-1)
            if buckets.numel() != batch_size:
                raise ValueError("bucket_ids must contain one id per batch item")
            if (
                int(buckets.min()) < 0
                or int(buckets.max()) >= self.positive_bank.num_buckets
            ):
                raise ValueError("bucket_ids are out of range")
        self.positive_bank.add(target_latent.detach(), buckets)
        positive = self.positive_bank.sample(
            buckets,
            min(batch_size, self.positive_bank.max_size),
            output_device=lq_latent_hat.device,
        )

        negative_counts = torch.tensor(
            [self.negative_bank.available(int(bucket)) for bucket in buckets],
            dtype=torch.long,
        )
        negative_count = int(negative_counts.min())
        if negative_count > 0:
            negative = self.negative_bank.sample(
                buckets,
                min(max(batch_size, 1), negative_count),
                output_device=lq_latent_hat.device,
            )
            negative_weight = torch.full(
                (batch_size, negative.shape[1]),
                self.negative_weight,
                device=lq_latent_hat.device,
            )
        else:
            negative = target_latent.new_zeros((batch_size, 0, *target_latent.shape[1:]))
            negative_weight = None

        optimizer.zero_grad(set_to_none=True)
        generated = self.generator(lq_latent_hat, ctxs, noise=noise)
        drift, _ = latent_drift_loss(
            generated.unsqueeze(1),
            positive,
            fixed_neg=negative,
            weight_neg=negative_weight,
            R_list=self.R_list,
        )
        reconstruction = F.mse_loss(generated, target_latent)
        drift_total = drift.mean() * self.positive_weight
        loss = drift_total + self.reconstruction_weight * reconstruction
        if not torch.isfinite(loss):
            raise FloatingPointError("non-finite drift training loss")
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.generator.parameters(),
            max_grad_norm,
        )
        optimizer.step()

        self.negative_bank.add(generated.detach(), buckets)
        return {
            "loss": float(loss.detach().cpu()),
            "drift": float(drift.mean().detach().cpu()),
            "reconstruction": float(reconstruction.detach().cpu()),
            "grad_norm": float(torch.as_tensor(grad_norm).detach().cpu()),
        }

    def clear(self) -> None:
        self.positive_bank.clear()
        self.negative_bank.clear()
