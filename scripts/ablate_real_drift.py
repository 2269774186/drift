#!/usr/bin/env python3
"""Tune the YODA-conditioned Drifting generator on cached real latents.

This is a held-out latent diagnostic, not a substitute for full-video
evaluation. Frames before ``--validation-start`` are used for training and
frames at or after that index are used for validation.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM_ROOT = ROOT / "upstream/YODA-main"
for path in (ROOT, UPSTREAM_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from yoda_drift.drift_model import ConditionalDriftGenerator
from yoda_drift.trainer import ConditionalDriftTrainer
from yoda_drift.yoda_hybrid import YodaHybridCodec


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache",
        type=Path,
        default=ROOT / "outputs/real_akiyo_yoda_drift/artifacts/training_cache.pt",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs/real_akiyo_yoda_drift_tuning",
    )
    parser.add_argument(
        "--sd-path",
        type=Path,
        default=ROOT / "checkpoints/sana-sprint",
    )
    parser.add_argument(
        "--inter-checkpoint",
        type=Path,
        default=ROOT / "checkpoints/yoda/YODA_inter_model_bpp2.pth",
    )
    parser.add_argument("--validation-start", type=int, default=180)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--configs",
        default="stable_rec5,stable_rec20,stable_rec100,small_rec20,legacy_like",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def json_dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=True, allow_nan=False),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def select_configs(names: str) -> list[dict[str, Any]]:
    available = {
        "stable_rec5": {
            "lr": 1e-4,
            "weight_decay": 0.0,
            "reconstruction_weight": 5.0,
            "hidden_channels": 64,
            "depth": 2,
            "max_grad_norm": 1.0,
        },
        "stable_rec20": {
            "lr": 1e-4,
            "weight_decay": 0.0,
            "reconstruction_weight": 20.0,
            "hidden_channels": 64,
            "depth": 2,
            "max_grad_norm": 1.0,
        },
        "stable_rec100": {
            "lr": 1e-4,
            "weight_decay": 0.0,
            "reconstruction_weight": 100.0,
            "hidden_channels": 64,
            "depth": 2,
            "max_grad_norm": 1.0,
        },
        "small_rec20": {
            "lr": 2e-4,
            "weight_decay": 0.0,
            "reconstruction_weight": 20.0,
            "hidden_channels": 32,
            "depth": 1,
            "max_grad_norm": 1.0,
        },
        "legacy_like": {
            "lr": 8e-4,
            "weight_decay": 1e-4,
            "reconstruction_weight": 20.0,
            "hidden_channels": 128,
            "depth": 3,
            "max_grad_norm": 2.0,
        },
    }
    selected = []
    for name in names.split(","):
        name = name.strip()
        if not name:
            continue
        if name not in available:
            raise ValueError(f"unknown config {name!r}")
        selected.append({"name": name, **available[name]})
    return selected


def load_model(args: argparse.Namespace, device: torch.device) -> YodaHybridCodec:
    model = YodaHybridCodec.from_pretrained(
        args.sd_path,
        args.inter_checkpoint,
        denoiser="drift",
        upstream_root=UPSTREAM_ROOT,
        device=device,
    )
    model.eval()
    model.requires_grad_(False)
    return model


def precompute_contexts(
    model: YodaHybridCodec,
    cache: list[dict[str, Any]],
    device: torch.device,
    batch_size: int,
) -> list[list[torch.Tensor]]:
    result: list[list[torch.Tensor]] = []
    with torch.inference_mode():
        for start in range(0, len(cache), batch_size):
            records = cache[start : start + batch_size]
            images = torch.stack([record["last_recon"].float() for record in records])
            images = images.to(device)
            features = model.vae.decoder.feature_extractor(images)
            result.extend(
                [
                    [value[index].half().cpu() for value in features]
                    for index in range(len(records))
                ]
            )
    torch.cuda.empty_cache()
    return result


def gather_batch(
    cache: list[dict[str, Any]],
    contexts: list[list[torch.Tensor]],
    indices: list[int],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor], torch.Tensor]:
    conditions = torch.stack([cache[index]["condition"].float() for index in indices])
    targets = torch.stack([cache[index]["target"].float() for index in indices])
    batch_contexts = [
        torch.stack(
            [contexts[index][scale].float() for index in indices]
        ).to(device)
        for scale in range(len(contexts[indices[0]]))
    ]
    frames = torch.tensor(
        [int(cache[index]["frame"]) for index in indices],
        dtype=torch.long,
    )
    return conditions.to(device), targets.to(device), batch_contexts, frames


@torch.inference_mode()
def evaluate(
    generator: ConditionalDriftGenerator,
    cache: list[dict[str, Any]],
    contexts: list[list[torch.Tensor]],
    indices: list[int],
    device: torch.device,
    batch_size: int,
) -> dict[str, float]:
    generator.eval()
    squared_error = 0.0
    absolute_error = 0.0
    identity_squared_error = 0.0
    target_squared = 0.0
    output_squared = 0.0
    residual_squared = 0.0
    samples = 0
    for start in range(0, len(indices), batch_size):
        batch_indices = indices[start : start + batch_size]
        condition, target, batch_contexts, _ = gather_batch(
            cache, contexts, batch_indices, device
        )
        generated = generator(condition, batch_contexts, noise=None)
        if not torch.isfinite(generated).all():
            raise FloatingPointError("non-finite validation latent")
        residual = generated - condition
        squared_error += float((generated - target).square().sum().cpu())
        absolute_error += float((generated - target).abs().sum().cpu())
        identity_squared_error += float((condition - target).square().sum().cpu())
        target_squared += float(target.square().sum().cpu())
        output_squared += float(generated.square().sum().cpu())
        residual_squared += float(residual.square().sum().cpu())
        samples += target.numel()
    generator.train()
    mse = squared_error / samples
    target_rms = (target_squared / samples) ** 0.5
    output_rms = (output_squared / samples) ** 0.5
    residual_rms = (residual_squared / samples) ** 0.5
    return {
        "mse": mse,
        "mae": absolute_error / samples,
        "psnr": -10.0 * torch.log10(torch.tensor(max(mse, 1e-12))).item(),
        "identity_mse": identity_squared_error / samples,
        "target_rms": target_rms,
        "output_rms": output_rms,
        "residual_rms": residual_rms,
    }


def run_config(
    config: dict[str, Any],
    model: YodaHybridCodec,
    cache: list[dict[str, Any]],
    contexts: list[list[torch.Tensor]],
    train_indices: list[int],
    validation_indices: list[int],
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, Any]:
    set_seed(args.seed)
    latent_shape = tuple(cache[0]["target"].shape)
    generator = ConditionalDriftGenerator(
        latent_channels=int(latent_shape[0]),
        context_channels=(1024, 512, 512, 256, 128),
        hidden_channels=int(config["hidden_channels"]),
        depth=int(config["depth"]),
        noise_scale=0.0,
    ).to(device)
    trainer = ConditionalDriftTrainer(
        generator,
        latent_shape=latent_shape,
        num_buckets=max(int(record["frame"]) for record in cache) + 1,
        max_positive=16,
        max_negative=16,
        positive_weight=1.0,
        negative_weight=1.0,
        reconstruction_weight=float(config["reconstruction_weight"]),
        R_list=(0.2, 0.05, 0.02),
        bank_device="cpu",
    )
    optimizer = torch.optim.AdamW(
        generator.parameters(),
        lr=float(config["lr"]),
        weight_decay=float(config["weight_decay"]),
    )
    rng = torch.Generator().manual_seed(args.seed)
    updates: list[dict[str, Any]] = []
    evaluations: list[dict[str, Any]] = []
    started = time.perf_counter()
    for step in range(args.steps + 1):
        if step in (0, 25, 50, 100, 200, 300, 400) or step == args.steps:
            metrics = evaluate(
                generator,
                cache,
                contexts,
                validation_indices,
                device,
                args.batch_size,
            )
            metrics["step"] = step
            evaluations.append(metrics)
            print(
                f"config={config['name']:13s} eval step={step:04d} "
                f"mse={metrics['mse']:.6g} psnr={metrics['psnr']:.3f} "
                f"residual_rms={metrics['residual_rms']:.4g}",
                flush=True,
            )
        if step == args.steps:
            break
        indices = [
            train_indices[int(index)]
            for index in torch.randint(
                len(train_indices),
                (min(args.batch_size, len(train_indices)),),
                generator=rng,
            ).tolist()
        ]
        condition, target, batch_contexts, frames = gather_batch(
            cache, contexts, indices, device
        )
        metrics = trainer.step(
            condition,
            target,
            batch_contexts,
            optimizer,
            noise=None,
            bucket_ids=frames,
            max_grad_norm=float(config["max_grad_norm"]),
        )
        metrics["step"] = step
        metrics["frames"] = frames.tolist()
        updates.append(metrics)
        if step == 0 or (step + 1) % 25 == 0:
            print(
                f"config={config['name']:13s} train step={step + 1:04d} "
                f"loss={metrics['loss']:.6g} drift={metrics['drift']:.6g} "
                f"recon={metrics['reconstruction']:.6g} "
                f"grad={metrics['grad_norm']:.4g}",
                flush=True,
            )
    elapsed = time.perf_counter() - started
    generator.eval()
    checkpoint = args.output_dir / f"{config['name']}.pt"
    torch.save(generator.state_dict(), checkpoint)
    return {
        "config": config,
        "train_records": len(train_indices),
        "validation_records": len(validation_indices),
        "train_frame_min": int(cache[train_indices[0]]["frame"]),
        "train_frame_max": int(cache[train_indices[-1]]["frame"]),
        "validation_frame_min": int(cache[validation_indices[0]]["frame"]),
        "validation_frame_max": int(cache[validation_indices[-1]]["frame"]),
        "elapsed_seconds": elapsed,
        "evaluations": evaluations,
        "updates": updates,
        "checkpoint": str(checkpoint),
    }


def main() -> None:
    args = parse_args()
    if args.steps <= 0:
        raise ValueError("--steps must be positive")
    if not 0 < args.validation_start < 200:
        raise ValueError("--validation-start must be between 1 and 199")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    cache = torch.load(args.cache, map_location="cpu", weights_only=False)
    cache = sorted(cache, key=lambda record: int(record["frame"]))
    train_indices = [
        index
        for index, record in enumerate(cache)
        if int(record["frame"]) < args.validation_start
    ]
    validation_indices = [
        index
        for index, record in enumerate(cache)
        if int(record["frame"]) >= args.validation_start
    ]
    if not train_indices or not validation_indices:
        raise RuntimeError("cache split produced an empty train or validation set")

    model = load_model(args, device)
    contexts = precompute_contexts(model, cache, device, args.batch_size)
    results = []
    for config in select_configs(args.configs):
        result = run_config(
            config,
            model,
            cache,
            contexts,
            train_indices,
            validation_indices,
            args,
            device,
        )
        results.append(result)
        json_dump(
            args.output_dir / "tuning.json",
            {
                "scope": "latent-only held-out tuning; not a video result",
                "cache": str(args.cache),
                "validation_start": args.validation_start,
                "steps": args.steps,
                "results": results,
            },
        )


if __name__ == "__main__":
    main()
