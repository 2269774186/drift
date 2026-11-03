"""A one-forward conditional generator for YODA latent refinement."""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class _ContextProjector(nn.Module):
    """Project one TA-AE context scale to the latent resolution."""

    def __init__(self, in_channels: int, latent_channels: int) -> None:
        super().__init__()
        hidden_channels = max(latent_channels, min(in_channels, latent_channels * 4))
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=1, bias=False),
            nn.GroupNorm(_groups(hidden_channels), hidden_channels),
            nn.SiLU(),
            nn.Conv2d(hidden_channels, latent_channels, kernel_size=3, padding=1),
        )

    def forward(self, ctx: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
        if ctx.ndim != 4:
            raise ValueError("each context tensor must have shape (B, C, H, W)")
        if ctx.shape[-2:] != size:
            ctx = F.interpolate(ctx, size=size, mode="bilinear", align_corners=False)
        return self.layers(ctx)


class ConditionalDriftGenerator(nn.Module):
    """Map compressed YODA latents to denoised latents in a single forward pass.

    This replaces the SANA DiT and its trainable LoRA adapters. The compressed
    latent is the primary condition; available TA-AE context maps are projected
    and added as spatial priors. ``noise`` is injected once for stochastic
    training and deterministic inference can omit it.

    Args:
        latent_channels: Number of compressed latent channels (YODA uses 32).
        context_channels: TA-AE context channels in feature-extractor order.
        hidden_channels: Width of the generator trunk.
        depth: Number of residual convolution blocks.
        noise_scale: Scale of optional training noise.
    """

    def __init__(
        self,
        latent_channels: int = 32,
        context_channels: Sequence[int] = (1024, 512, 512, 256, 128),
        hidden_channels: int = 128,
        depth: int = 3,
        noise_scale: float = 0.05,
    ) -> None:
        super().__init__()
        if latent_channels <= 0 or hidden_channels <= 0 or depth <= 0:
            raise ValueError("latent_channels, hidden_channels and depth must be positive")
        if noise_scale < 0:
            raise ValueError("noise_scale must be non-negative")

        self.latent_channels = int(latent_channels)
        self.context_channels = tuple(int(v) for v in context_channels)
        self.noise_scale = float(noise_scale)

        self.input_projection = nn.Conv2d(self.latent_channels, hidden_channels, 3, padding=1)
        self.context_projections = nn.ModuleList(
            _ContextProjector(channels, hidden_channels)
            for channels in self.context_channels
        )
        self.context_scale = nn.Parameter(torch.zeros(max(len(self.context_channels), 1)))
        self.blocks = nn.ModuleList(
            _ResidualBlock(hidden_channels) for _ in range(int(depth))
        )
        self.output_projection = nn.Conv2d(hidden_channels, self.latent_channels, 3, padding=1)
        nn.init.zeros_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)

    def forward(
        self,
        lq_latent_hat: torch.Tensor,
        ctxs: Sequence[torch.Tensor] | None = None,
        noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return refined latents with the same shape as ``lq_latent_hat``."""
        if lq_latent_hat.ndim != 4:
            raise ValueError("lq_latent_hat must have shape (B, C, H, W)")
        if lq_latent_hat.shape[1] != self.latent_channels:
            raise ValueError(
                f"expected {self.latent_channels} latent channels, "
                f"got {lq_latent_hat.shape[1]}"
            )
        if ctxs is not None and len(ctxs) != len(self.context_channels):
            raise ValueError(
                f"expected {len(self.context_channels)} context scales, got {len(ctxs)}"
            )

        latent = lq_latent_hat
        if noise is not None and self.noise_scale:
            if noise.shape != lq_latent_hat.shape:
                raise ValueError("noise must have the same shape as lq_latent_hat")
            latent = latent + self.noise_scale * noise.to(latent.dtype)

        hidden = self.input_projection(latent)
        if ctxs is not None:
            size = (latent.shape[-2], latent.shape[-1])
            for index, (projection, ctx) in enumerate(
                zip(self.context_projections, ctxs)
            ):
                if ctx.shape[0] != latent.shape[0]:
                    raise ValueError(
                        "context and latent batch sizes must match, "
                        f"got {ctx.shape[0]} and {latent.shape[0]}"
                    )
                hidden = hidden + self.context_scale[index] * projection(ctx, size)
        for block in self.blocks:
            hidden = block(hidden)
        return lq_latent_hat + self.output_projection(hidden)


class _ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        groups = _groups(channels)
        self.norm1 = nn.GroupNorm(groups, channels)
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(groups, channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.conv1(F.silu(self.norm1(x)))
        x = self.conv2(F.silu(self.norm2(x)))
        return residual + x


def _groups(channels: int) -> int:
    for groups in (32, 16, 8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1
