"""Run a small, reproducible YODA-conditioned Drifting smoke experiment.

This script isolates the latent refinement stage. It does not claim to reproduce
YODA's paper metrics because the released YODA checkpoint and entropy coder are
not required here; instead it verifies one-forward conditioning, memory-bank
training and reconstruction on synthetic sequential latents.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import sys
import time
from pathlib import Path

import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from yoda_drift.drift_model import ConditionalDriftGenerator
from yoda_drift.trainer import ConditionalDriftTrainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--frames", type=int, default=8)
    parser.add_argument("--latent-size", type=int, default=8)
    parser.add_argument("--latent-channels", type=int, default=4)
    parser.add_argument("--context-channels", type=int, nargs="+", default=[8, 4])
    parser.add_argument("--hidden-channels", type=int, default=16)
    parser.add_argument("--depth", type=int, default=1)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/drift_smoke"))
    return parser.parse_args()


def make_sequence(
    frames: int,
    batch_size: int,
    channels: int,
    size: int,
    generator: torch.Generator,
) -> torch.Tensor:
    """Create smooth motion plus a small amount of independent texture."""
    base = torch.randn(batch_size, channels, size, size, generator=generator)
    sequence = []
    for frame in range(frames):
        phase = 2.0 * math.pi * frame / max(frames, 1)
        shift = int(round(math.sin(phase) * max(size // 8, 1)))
        shifted = torch.roll(base, shifts=shift, dims=-1)
        texture = 0.05 * torch.randn(
            batch_size,
            channels,
            size,
            size,
            generator=generator,
        )
        sequence.append(torch.tanh(shifted + texture))
    return torch.stack(sequence)


def make_context(
    target: torch.Tensor,
    channels: list[int],
    generator: torch.Generator,
) -> list[torch.Tensor]:
    """Stand in for TA-AE multiscale reference-frame features."""
    batch, _, height, width = target.shape
    context = []
    for index, channel_count in enumerate(channels):
        scale = 2**index
        size = (max(height // scale, 1), max(width // scale, 1))
        base = target.mean(dim=1, keepdim=True)
        base = torch.nn.functional.interpolate(
            base,
            size=size,
            mode="bilinear",
            align_corners=False,
        )
        noise = 0.05 * torch.randn(
            batch,
            channel_count,
            *size,
            generator=generator,
        )
        context.append(base.repeat(1, channel_count, 1, 1) + noise)
    return context


def latent_to_preview(
    condition: torch.Tensor,
    target: torch.Tensor,
    generated: torch.Tensor,
) -> Image.Image:
    """Write a compact RGB preview of the first sample's latent state."""
    rows = []
    for name, latent in (
        ("condition", condition),
        ("target", target),
        ("generated", generated),
    ):
        image = latent[0].mean(dim=0).detach().cpu()
        image = image.sub(image.min()).div(image.max() - image.min() + 1e-8)
        image = (image * 255).round().to(torch.uint8)
        rgb = Image.fromarray(image.numpy()).convert("RGB")
        rows.append((name, rgb))

    scale = 8
    label_width = 72
    panel_width = rows[0][1].width * scale
    panel_height = rows[0][1].height * scale
    canvas = Image.new(
        "RGB",
        (label_width + panel_width, panel_height * len(rows)),
        color=(255, 255, 255),
    )
    from PIL import ImageDraw

    draw = ImageDraw.Draw(canvas)
    for row_index, (name, image) in enumerate(rows):
        canvas.paste(
            image.resize((panel_width, panel_height), resample=Image.Resampling.NEAREST),
            (label_width, row_index * panel_height),
        )
        draw.text((4, row_index * panel_height + 4), name, fill=(0, 0, 0))
    return canvas


def run(args: argparse.Namespace) -> dict[str, object]:
    if args.steps <= 0 or args.batch_size <= 0 or args.frames <= 0:
        raise ValueError("steps, batch-size and frames must be positive")
    if args.latent_size <= 0 or args.latent_channels <= 0 or args.hidden_channels <= 0:
        raise ValueError("latent and hidden dimensions must be positive")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")

    torch.manual_seed(args.seed)
    generator = torch.Generator().manual_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    sequence = make_sequence(
        args.frames,
        args.batch_size,
        args.latent_channels,
        args.latent_size,
        generator,
    )
    # A frame's previous latent is its condition; a fixed offset is the target.
    condition = sequence[:-1]
    target = sequence[1:]
    ctxs = [
        make_context(frame, args.context_channels, generator)
        for frame in condition
    ]

    model = ConditionalDriftGenerator(
        latent_channels=args.latent_channels,
        context_channels=args.context_channels,
        hidden_channels=args.hidden_channels,
        depth=args.depth,
    ).to(device)
    trainer = ConditionalDriftTrainer(
        model,
        latent_shape=target.shape[2:],
        max_positive=16,
        max_negative=16,
        reconstruction_weight=0.1,
        R_list=(0.2, 0.05, 0.02),
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    losses = []
    grad_norms = []
    started = time.perf_counter()
    model.train()
    for step in range(args.steps):
        index = step % (args.frames - 1)
        metrics = trainer.step(
            condition[index].to(device),
            target[index].to(device),
            [value.to(device) for value in ctxs[index]],
            optimizer,
        )
        losses.append(metrics)
        grad_norms.append(metrics["grad_norm"])
        if step == 0 or (step + 1) % max(args.steps // 10, 1) == 0:
            print(f"step={step + 1:4d} " + " ".join(
                f"{key}={value:.6g}" for key, value in metrics.items()
            ))
    elapsed = time.perf_counter() - started

    model.eval()
    with torch.no_grad():
        generated = model(condition[-1].to(device), [
            value.to(device) for value in ctxs[-1]
        ]).cpu()
    generated_mse = float(
        torch.nn.functional.mse_loss(generated, target[-1])
    )
    preview = latent_to_preview(condition[-1], target[-1], generated)
    preview.save(args.output_dir / "latent_preview.png")
    torch.save(model.state_dict(), args.output_dir / "generator.pt")

    initial = sum(item["loss"] for item in losses[:5]) / min(5, len(losses))
    final = sum(item["loss"] for item in losses[-5:]) / min(5, len(losses))
    result: dict[str, object] = {
        "status": "ok",
        "scope": "synthetic latent smoke; not paper metric reproduction",
        "config": {
            **vars(args),
            "output_dir": str(args.output_dir),
        },
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "device": str(device),
        },
        "metrics": {
            "steps": args.steps,
            "initial_loss_mean": initial,
            "final_loss_mean": final,
            "loss_ratio": final / max(initial, 1e-12),
            "elapsed_seconds": elapsed,
            "steps_per_second": args.steps / max(elapsed, 1e-12),
            "max_grad_norm": max(grad_norms),
            "final_generated_mse": generated_mse,
            "peak_cuda_memory_mb": (
                torch.cuda.max_memory_allocated(device) / (1024**2)
                if device.type == "cuda"
                else 0.0
            ),
        },
        "history": losses,
    }
    (args.output_dir / "metrics.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )
    print(json.dumps(result["metrics"], indent=2))
    return result


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
