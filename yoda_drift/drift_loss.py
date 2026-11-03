"""PyTorch port of the drift objective released with arXiv:2602.04770."""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F


def _pairwise_distance(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    xydot = torch.einsum("bnd,bmd->bnm", x, y)
    x_norms = torch.einsum("bnd,bnd->bn", x, x)
    y_norms = torch.einsum("bmd,bmd->bm", y, y)
    sq_dist = x_norms[:, :, None] + y_norms[:, None, :] - 2.0 * xydot
    return torch.sqrt(sq_dist.clamp_min(eps))


def drift_loss(
    gen: torch.Tensor,
    fixed_pos: torch.Tensor,
    fixed_neg: torch.Tensor | None = None,
    weight_gen: torch.Tensor | None = None,
    weight_pos: torch.Tensor | None = None,
    weight_neg: torch.Tensor | None = None,
    R_list: Sequence[float] = (0.2, 0.05, 0.02),
    eps: float = 1e-8,
    return_info: bool = True,
) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute the multi-temperature kernel drift objective.

    Args:
        gen: Generated features with shape ``(B, G, D)``.
        fixed_pos: Positive/real features with shape ``(B, P, D)``.
        fixed_neg: Negative/generated-history features with shape ``(B, N, D)``.
        weight_gen: Optional per-generated-sample weights, shape ``(B, G)``.
        weight_pos: Optional positive weights, shape ``(B, P)``.
        weight_neg: Optional negative weights, shape ``(B, N)``.
        R_list: Kernel temperatures. The released training config uses
            ``(0.2, 0.05, 0.02)``.
        eps: Minimum squared distance used by ``cdist``.
        return_info: Return per-temperature diagnostics alongside the loss.

    Returns:
        A per-batch loss with shape ``(B,)`` and, optionally, a diagnostics
        dictionary. The drift goal is detached, matching the official stop-gradient
        implementation.
    """
    if gen.ndim != 3 or fixed_pos.ndim != 3:
        raise ValueError("gen and fixed_pos must have shape (B, N, D)")
    if gen.shape[0] != fixed_pos.shape[0] or gen.shape[2] != fixed_pos.shape[2]:
        raise ValueError("gen and fixed_pos must agree on batch and feature dimensions")

    batch_size, gen_count, feature_dim = gen.shape
    pos_count = fixed_pos.shape[1]
    if fixed_neg is None:
        fixed_neg = gen.new_zeros((batch_size, 0, feature_dim))
    if fixed_neg.ndim != 3 or fixed_neg.shape[0] != batch_size or fixed_neg.shape[2] != feature_dim:
        raise ValueError("fixed_neg has incompatible shape")
    neg_count = fixed_neg.shape[1]

    gen = gen.float()
    fixed_pos = fixed_pos.float()
    fixed_neg = fixed_neg.float()
    weight_gen = _weights_or_ones(weight_gen, batch_size, gen_count, gen.device)
    weight_pos = _weights_or_ones(weight_pos, batch_size, pos_count, gen.device)
    weight_neg = _weights_or_ones(weight_neg, batch_size, neg_count, gen.device)

    with torch.no_grad():
        old_gen = gen.detach()
        targets = torch.cat((old_gen, fixed_neg.detach(), fixed_pos.detach()), dim=1)
        target_weights = torch.cat((weight_gen.detach(), weight_neg.detach(), weight_pos.detach()), dim=1)

        dist = _pairwise_distance(old_gen, targets, eps=eps)
        weighted_dist = dist * target_weights[:, None, :]
        scale = weighted_dist.mean() / target_weights.mean().clamp_min(eps)

        token_count = max(feature_dim, 1)
        scale_inputs = (scale / token_count**0.5).clamp_min(1e-3)
        old_gen_scaled = old_gen / scale_inputs
        targets_scaled = targets / scale_inputs
        dist_normed = dist / scale.clamp_min(1e-3)

        diagonal_mask = torch.eye(gen_count, dtype=torch.float32, device=gen.device)
        block_mask = F.pad(diagonal_mask, (0, neg_count + pos_count))
        dist_normed = dist_normed + block_mask.unsqueeze(0) * 100.0

        force_across_temperatures = torch.zeros_like(old_gen_scaled)
        info: dict[str, torch.Tensor] = {"scale": scale.detach()}
        for temperature in R_list:
            if temperature <= 0:
                raise ValueError("all kernel temperatures must be positive")
            logits = -dist_normed / float(temperature)
            affinity = torch.softmax(logits, dim=-1)
            affinity_transpose = torch.softmax(logits, dim=-2)
            affinity = torch.sqrt((affinity * affinity_transpose).clamp_min(1e-6))
            affinity = affinity * target_weights[:, None, :]

            split_index = gen_count + neg_count
            negative_affinity = affinity[:, :, :split_index]
            positive_affinity = affinity[:, :, split_index:]

            positive_mass = positive_affinity.sum(dim=-1, keepdim=True)
            negative_coeff = -negative_affinity * positive_mass
            negative_mass = negative_affinity.sum(dim=-1, keepdim=True)
            positive_coeff = positive_affinity * negative_mass
            coefficients = torch.cat((negative_coeff, positive_coeff), dim=2)

            total_force = torch.bmm(coefficients, targets_scaled)
            total_coefficients = coefficients.sum(dim=-1)
            total_force = total_force - total_coefficients[..., None] * old_gen_scaled
            force_norm = total_force.square().mean()
            info[f"loss_{temperature:g}"] = force_norm.detach()
            force_scale = force_norm.clamp_min(1e-8).sqrt()
            force_across_temperatures = force_across_temperatures + total_force / force_scale

        goal_scaled = old_gen_scaled + force_across_temperatures

    gen_scaled = gen / scale_inputs
    loss = (gen_scaled - goal_scaled).square().mean(dim=(-1, -2))
    if not return_info:
        return loss
    info = {name: value.detach().mean() for name, value in info.items()}
    return loss, info


def latent_drift_loss(
    gen: torch.Tensor,
    fixed_pos: torch.Tensor,
    fixed_neg: torch.Tensor | None = None,
    weight_gen: torch.Tensor | None = None,
    weight_pos: torch.Tensor | None = None,
    weight_neg: torch.Tensor | None = None,
    R_list: Sequence[float] = (0.2, 0.05, 0.02),
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Apply drift loss independently to every spatial token of latent maps.

    Args:
        gen: Generated latents with shape ``(B, G, C, H, W)``.
        fixed_pos: Positive latents with shape ``(B, P, C, H, W)``.
        fixed_neg: Negative latents with shape ``(B, N, C, H, W)``.
        weight_gen: Optional per-sample weights ``(B, G)``.
        weight_pos: Optional positive weights ``(B, P)``.
        weight_neg: Optional negative weights ``(B, N)``.
    """
    if gen.ndim != 5 or fixed_pos.ndim != 5:
        raise ValueError("gen and fixed_pos must have shape (B, N, C, H, W)")
    if gen.shape[0] != fixed_pos.shape[0]:
        raise ValueError("gen and fixed_pos must have the same batch size")
    if gen.shape[2:] != fixed_pos.shape[2:]:
        raise ValueError("gen and fixed_pos must have the same latent shape")
    if fixed_neg is not None and (
        fixed_neg.ndim != 5
        or fixed_neg.shape[0] != gen.shape[0]
        or fixed_neg.shape[2:] != gen.shape[2:]
    ):
        raise ValueError("fixed_neg has incompatible shape")

    gen_tokens = _to_token_sets(gen)
    pos_tokens = _to_token_sets(fixed_pos)
    neg_tokens = None if fixed_neg is None else _to_token_sets(fixed_neg)
    token_loss, info = drift_loss(
        gen_tokens,
        pos_tokens,
        fixed_neg=neg_tokens,
        weight_gen=_token_weights(weight_gen, gen.shape),
        weight_pos=_token_weights(weight_pos, fixed_pos.shape),
        weight_neg=None
        if fixed_neg is None
        else _token_weights(weight_neg, fixed_neg.shape),
        R_list=R_list,
    )
    batch_size, token_count = gen.shape[0], gen.shape[-2] * gen.shape[-1]
    return token_loss.reshape(batch_size, token_count).mean(dim=1), info


def _to_token_sets(samples: torch.Tensor) -> torch.Tensor:
    batch_size, sample_count, channels, height, width = samples.shape
    # (B, N, C, H, W) -> (B * H * W, N, C)
    return samples.permute(0, 3, 4, 2, 1).reshape(batch_size * height * width, sample_count, channels)


def _token_weights(
    weights: torch.Tensor | None,
    samples_shape: torch.Size,
) -> torch.Tensor | None:
    if weights is None:
        return None
    if weights.ndim != 2 or weights.shape != samples_shape[:2]:
        raise ValueError(
            f"expected weights with shape {tuple(samples_shape[:2])}, "
            f"got {tuple(weights.shape)}"
        )
    _, sample_count, _, height, width = samples_shape
    return (
        weights[:, :, None, None]
        .expand(weights.shape[0], sample_count, height, width)
        .permute(0, 2, 3, 1)
        .reshape(weights.shape[0] * height * width, sample_count)
    )


def _weights_or_ones(
    weights: torch.Tensor | None,
    batch_size: int,
    count: int,
    device: torch.device,
) -> torch.Tensor:
    if weights is None:
        return torch.ones((batch_size, count), dtype=torch.float32, device=device)
    if weights.shape != (batch_size, count):
        raise ValueError(f"expected weight shape {(batch_size, count)}, got {tuple(weights.shape)}")
    return weights.float()
