#!/usr/bin/env python3
"""Compare how YODA decodes target, quantized, and Drift-generated latents."""

from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
UPSTREAM_ROOT = ROOT / "upstream/YODA-main"
if str(UPSTREAM_ROOT) not in sys.path:
    sys.path.insert(0, str(UPSTREAM_ROOT))

from yoda_drift.drift_model import ConditionalDriftGenerator
from yoda_drift.yoda_hybrid import YodaHybridCodec


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=ROOT / "outputs/real_akiyo_yoda_drift_v2",
    )
    parser.add_argument(
        "--sd-path",
        type=Path,
        default=ROOT / "checkpoints/sana-sprint",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=ROOT / "checkpoints/yoda/YODA_inter_model_bpp2.pth",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--sample-stride", type=int, default=20)
    return parser.parse_args()


def psnr_per_image(reference: torch.Tensor, reconstructed: torch.Tensor) -> torch.Tensor:
    mse = (reference - reconstructed).square().flatten(1).mean(1)
    return 10.0 * torch.log10(1.0 / mse.clamp_min(1e-12))


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    device = torch.device(args.device)
    frame_cache = np.load(
        run_dir / "artifacts/frames.npy",
        mmap_mode="r",
    )
    records = torch.load(
        run_dir / "artifacts/training_cache.pt",
        map_location="cpu",
        weights_only=False,
    )
    history = json.loads(
        (run_dir / "train_history.json").read_text(encoding="utf-8")
    )

    model = YodaHybridCodec.from_pretrained(
        args.sd_path,
        args.checkpoint,
        denoiser="drift",
        upstream_root=UPSTREAM_ROOT,
        device=device,
    )
    model.eval()
    model.requires_grad_(False)
    generator = ConditionalDriftGenerator(
        latent_channels=int(records[0]["target"].shape[0]),
        context_channels=(1024, 512, 512, 256, 128),
        hidden_channels=int(history.get("hidden_channels", 128)),
        depth=int(history.get("depth", 3)),
        noise_scale=float(history.get("noise_scale", 0.0)),
    ).to(device)
    generator.load_state_dict(
        torch.load(
            run_dir / "artifacts/generator.pt",
            map_location=device,
            weights_only=True,
        )
    )
    generator.eval()

    selected = list(range(0, len(records), max(args.sample_stride, 1)))
    summaries: dict[str, list[float]] = {
        "condition": [],
        "target": [],
        "generated": [],
    }
    latent_metrics: dict[str, list[float]] = {
        "condition_to_target_mse": [],
        "generated_to_target_mse": [],
        "generated_to_condition_mse": [],
    }
    with torch.inference_mode():
        for start in range(0, len(selected), args.batch_size):
            items = [records[index] for index in selected[start : start + args.batch_size]]
            frame_ids = [int(item["frame"]) for item in items]
            condition = torch.stack([item["condition"].float() for item in items]).to(
                device
            )
            target = torch.stack([item["target"].float() for item in items]).to(device)
            context_source = torch.stack(
                [item["last_recon"].float() for item in items]
            ).to(device)
            if condition.ndim == 5:
                condition = condition[:, 0]
            if target.ndim == 5:
                target = target[:, 0]
            if context_source.ndim == 5:
                context_source = context_source[:, 0]

            contexts = model.vae.decoder.feature_extractor(context_source)
            generated = generator(condition, contexts, noise=None)
            reference_np = np.stack(
                [
                    cv2.cvtColor(np.asarray(frame_cache[index]), cv2.COLOR_BGR2RGB)
                    for index in frame_ids
                ]
            )
            reference = (
                torch.from_numpy(reference_np)
                .to(device)
                .permute(0, 3, 1, 2)
                .float()
                .div_(255.0)
            )
            decoded = {
                "condition": model._decode(condition, contexts, clamp=True),
                "target": model._decode(target, contexts, clamp=True),
                "generated": model._decode(generated, contexts, clamp=True),
            }
            for name, image in decoded.items():
                image = image[:, :, : reference.shape[-2], : reference.shape[-1]]
                image = (image + 1.0).mul_(0.5).clamp_(0.0, 1.0)
                summaries[name].extend(
                    psnr_per_image(reference, image).float().cpu().tolist()
                )
            latent_metrics["condition_to_target_mse"].append(
                float(F.mse_loss(condition, target).cpu())
            )
            latent_metrics["generated_to_target_mse"].append(
                float(F.mse_loss(generated, target).cpu())
            )
            latent_metrics["generated_to_condition_mse"].append(
                float(F.mse_loss(generated, condition).cpu())
            )
            print(
                "frames="
                + ",".join(str(value) for value in frame_ids)
                + " "
                + " ".join(
                    f"{name}_psnr={np.mean(summaries[name][-len(frame_ids):]):.4f}"
                    for name in summaries
                )
            )

    report = {
        "sample_count": len(selected),
        "frame_ids": selected,
        "decode_psnr": {
            name: {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
            }
            for name, values in summaries.items()
        },
        "latent_mse": {
            name: float(np.mean(values))
            for name, values in latent_metrics.items()
        },
    }
    output_path = run_dir / "diagnostics/training_latent_decode.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=True, allow_nan=False),
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2))
    del model, generator
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
