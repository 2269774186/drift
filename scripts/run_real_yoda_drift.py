#!/usr/bin/env python3
"""Run a complete YODA video comparison with a Drifting generator.

The experiment uses YODA's real intra/inter codecs and bitstream parser. The
first ``--train-frames`` frames are used only to train the conditional Drifting
generator; metrics and comparisons for the remaining frames are held out.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import io
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable

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
from yoda_drift.trainer import ConditionalDriftTrainer
from yoda_drift.yoda_hybrid import YodaHybridCodec


CHECKPOINTS = {
    "sana": (
        ROOT / "checkpoints/sana-sprint/transformer/diffusion_pytorch_model.safetensors",
        "9069dd697395b96a59d1f74837461f7aeb3d95e55b67fa85c9f31c9106b65e87",
    ),
    "yoda_inter": (
        ROOT / "checkpoints/yoda/YODA_inter_model_bpp2.pth",
        "1c8f9bfe379f4bda4f8221e60520bc7700be81e13a3256b1b3018c9534399ad0",
    ),
    "yoda_intra": (
        ROOT / "checkpoints/yoda/YODA_intra_model.pth",
        "8073bdef46a6ba37fe2e6de5b96c19cf8a8fc52e78190657e80478ece8c1c3cc",
    ),
}

MODE_LABELS = {
    "yoda_original": "YODA original",
    "drift_identity": "Drifting identity",
    "drift_trained": "Drifting trained",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "data/akiyo_cif.y4m")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs/real_akiyo_yoda_drift_v2",
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
    parser.add_argument(
        "--intra-checkpoint",
        type=Path,
        default=ROOT / "checkpoints/yoda/YODA_intra_model.pth",
    )
    parser.add_argument(
        "--prompt-embeds",
        type=Path,
        default=ROOT / "checkpoints/yoda/prompt_embeds.pt",
    )
    parser.add_argument(
        "--prompt-attention-mask",
        type=Path,
        default=ROOT / "checkpoints/yoda/prompt_attention_mask.pt",
    )
    parser.add_argument("--max-frames", type=int, default=300)
    parser.add_argument("--train-frames", type=int, default=200)
    parser.add_argument("--intra-period", type=int, default=32)
    parser.add_argument("--train-steps", type=int, default=600)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--reconstruction-weight", type=float, default=20.0)
    parser.add_argument("--max-positive", type=int, default=32)
    parser.add_argument("--max-negative", type=int, default=32)
    parser.add_argument("--hidden-channels", type=int, default=128)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--lr", type=float, default=8e-4)
    parser.add_argument("--grad-clip", type=float, default=2.0)
    parser.add_argument("--noise-scale", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lora-rank", type=int, default=72)
    parser.add_argument(
        "--precision",
        choices=("fp32", "bf16"),
        default="fp32",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--reuse", action="store_true")
    parser.add_argument("--keep-raw", action="store_true")
    parser.add_argument("--hash-cache-only", action="store_true")
    return parser.parse_args()


def json_dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=True, allow_nan=False),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(16 * 1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def verify_checkpoints(args: argparse.Namespace) -> dict[str, dict[str, Any]]:
    paths = {
        "sana": args.sd_path / "transformer/diffusion_pytorch_model.safetensors",
        "yoda_inter": args.inter_checkpoint,
        "yoda_intra": args.intra_checkpoint,
    }
    report: dict[str, dict[str, Any]] = {}
    for name, path in paths.items():
        expected_path, expected_hash = CHECKPOINTS[name]
        del expected_path
        if not path.is_file():
            raise FileNotFoundError(f"missing checkpoint: {path}")
        actual_size = path.stat().st_size
        actual_hash = sha256_file(path)
        report[name] = {
            "path": str(path),
            "size": actual_size,
            "sha256": actual_hash,
            "expected_sha256": expected_hash,
            "verified": actual_hash == expected_hash,
        }
        if actual_hash != expected_hash:
            raise RuntimeError(
                f"checkpoint hash mismatch for {name}: "
                f"expected {expected_hash}, got {actual_hash}"
            )
    return report


def set_seed(seed: int) -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":16:8")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


def tensor_to_bgr(image: torch.Tensor, height: int, width: int) -> np.ndarray:
    image = image[:1, :, :height, :width].detach().float().clamp(-1, 1)
    rgb = ((image[0].permute(1, 2, 0) + 1.0) * 127.5).round()
    rgb = rgb.to(torch.uint8).cpu().numpy()
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def bgr_to_model_input(
    frame: np.ndarray,
    device: torch.device,
    padding_right: int,
    padding_bottom: int,
) -> torch.Tensor:
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = (
        torch.from_numpy(rgb)
        .to(device=device, dtype=torch.float32)
        .permute(2, 0, 1)
        .unsqueeze(0)
        / 127.5
        - 1.0
    )
    if padding_right or padding_bottom:
        image = F.pad(
            image,
            (0, padding_right, 0, padding_bottom),
            mode="replicate",
        )
    return image.contiguous()


def load_frames(path: Path, max_frames: int) -> tuple[list[np.ndarray], float]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"could not open video: {path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frames: list[np.ndarray] = []
    while len(frames) < max_frames:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(frame)
    capture.release()
    if len(frames) != max_frames:
        raise RuntimeError(
            f"expected {max_frames} frames from {path}, decoded {len(frames)}"
        )
    if min(frame.shape[:2]) <= 0:
        raise RuntimeError("decoded an invalid video frame")
    return frames, fps


class FFmpegWriter:
    """Stream BGR frames to H.264 without staging large intermediate videos."""

    def __init__(self, path: Path, width: int, height: int, fps: float) -> None:
        self.path = path
        self.width = int(width)
        self.height = int(height)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.log_path = path.with_suffix(path.suffix + ".ffmpeg.log")
        self.log_handle = self.log_path.open("wb")
        command = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-s",
            f"{self.width}x{self.height}",
            "-r",
            f"{fps:.9f}",
            "-i",
            "-",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "16",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(path),
        ]
        self.process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=self.log_handle,
        )
        self.closed = False

    def write(self, frame: np.ndarray) -> None:
        if self.closed or self.process.stdin is None:
            raise RuntimeError("video writer is closed")
        if frame.shape != (self.height, self.width, 3) or frame.dtype != np.uint8:
            raise ValueError(
                f"expected ({self.height}, {self.width}, 3) uint8, got "
                f"{frame.shape} {frame.dtype}"
            )
        self.process.stdin.write(np.ascontiguousarray(frame).tobytes())

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        assert self.process.stdin is not None
        self.process.stdin.close()
        return_code = self.process.wait()
        self.log_handle.close()
        if return_code != 0:
            error = self.log_path.read_text(encoding="utf-8", errors="replace")
            raise RuntimeError(f"ffmpeg failed for {self.path}: {error}")
        if not self.path.is_file() or self.path.stat().st_size == 0:
            raise RuntimeError(f"ffmpeg produced no output for {self.path}")


def save_frames_cache(path: Path, frames: list[np.ndarray], reuse: bool) -> None:
    if reuse and path.is_file():
        return
    temporary = path.with_suffix(".tmp.npy")
    with temporary.open("wb") as handle:
        np.save(handle, np.stack(frames, axis=0))
    os.replace(temporary, path)


def load_frames_cache(path: Path) -> np.ndarray:
    frames = np.load(path, mmap_mode="r")
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise RuntimeError(f"invalid frame cache: {path} {frames.shape}")
    return frames


def padding(frame: np.ndarray) -> tuple[int, int]:
    height, width = frame.shape[:2]
    new_height = math.ceil(height / 64) * 64
    new_width = math.ceil(width / 64) * 64
    return new_width - width, new_height - height


def checkpoint_loader(path: Path) -> dict[str, Any]:
    from utils.utils import get_state_dict

    return get_state_dict(path)


def load_matching(module: torch.nn.Module, weights: dict[str, Any]) -> int:
    current = module.state_dict()
    matching = {
        key: value
        for key, value in weights.items()
        if key in current and value.shape == current[key].shape
    }
    module.load_state_dict(matching, strict=False)
    return len(matching)


def precompute_intra(
    args: argparse.Namespace,
    frames: np.ndarray,
    fps: float,
    output_dir: Path,
    device: torch.device,
) -> dict[str, Any]:
    cache_path = output_dir / "artifacts/intra_records.pt"
    if args.reuse and cache_path.is_file():
        return torch.load(cache_path, map_location="cpu", weights_only=False)

    from src.YODA_INTRA import Yoda_Intra

    intra_args = SimpleNamespace(
        timestep_i=999,
        prompt_embeds_path=str(args.prompt_embeds),
        prompt_attention_mask_path=str(args.prompt_attention_mask),
        lora_rank_transformer=args.lora_rank,
        pretrained_i_weights=str(args.intra_checkpoint),
    )
    model = Yoda_Intra(sd_path=str(args.sd_path), args=intra_args).to(device)
    model.codec.update()
    model.eval()
    model.transformer.requires_grad_(False)
    model.vae.requires_grad_(False)
    intra_checkpoint = checkpoint_loader(args.intra_checkpoint)
    report = {
        "codec": load_matching(
            model.codec,
            intra_checkpoint["state_dict_codec"],
        ),
        "vae": load_matching(
            model.vae,
            intra_checkpoint["state_dict_vae"],
        ),
        "transformer": load_matching(
            model.transformer,
            intra_checkpoint["state_dict_transformer"],
        ),
    }
    if report["transformer"] < 400:
        raise RuntimeError(f"intra LoRA matching is unexpectedly low: {report}")

    height, width = frames.shape[1:3]
    pad_right, pad_bottom = padding(frames[0])
    records: dict[int, dict[str, Any]] = {}
    with torch.inference_mode():
        for index in range(0, len(frames), args.intra_period):
            x_padded = bgr_to_model_input(
                frames[index],
                device,
                pad_right,
                pad_bottom,
            )
            sps = {
                "sps_id": 0,
                "height": height,
                "width": width,
                "ec_part": 0,
                "use_ada_i": 0,
            }
            torch.cuda.synchronize(device)
            started = time.perf_counter()
            encoded = model.compress(x_padded)
            torch.cuda.synchronize(device)
            encoded_seconds = time.perf_counter() - started
            torch.cuda.synchronize(device)
            started = time.perf_counter()
            decoded = model.decompress(encoded["bit_stream"], sps)
            torch.cuda.synchronize(device)
            decoded_seconds = time.perf_counter() - started
            mismatch = float(
                (decoded["x_hat"] - encoded["x_hat"]).abs().max().cpu()
            )
            if mismatch > 1e-3:
                raise RuntimeError(
                    f"intra entropy round trip mismatch at frame {index}: {mismatch}"
                )
            records[index] = {
                "bit_stream": bytes(encoded["bit_stream"]),
                "sps": sps,
                "x_hat": decoded["x_hat"].detach().cpu(),
                "x_denoised": decoded["x_denoised"].detach().cpu(),
                "encode_seconds": encoded_seconds,
                "decode_seconds": decoded_seconds,
                "roundtrip_max_abs": mismatch,
            }
            print(f"intra frame={index:03d} roundtrip={mismatch:.3e}")
    cache = {
        "fps": fps,
        "height": height,
        "width": width,
        "intra_period": args.intra_period,
        "load_report": report,
        "records": records,
    }
    torch.save(cache, cache_path)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return cache


def add_sps_and_packet(
    output: io.BytesIO,
    sps_helper: Any,
    bit_stream: bytes,
    *,
    is_i_frame: bool,
    sps: dict[str, Any],
) -> int:
    from src.rt_modules.utils.stream_helper import write_ip, write_sps

    before = output.tell()
    resolved_id, is_new = sps_helper.get_sps_id(sps)
    sps["sps_id"] = resolved_id
    if is_new:
        write_sps(output, sps)
    write_ip(output, is_i_frame, resolved_id, 0, bit_stream)
    return output.tell() - before


def write_bitstream_file(
    path: Path,
    records: list[dict[str, Any]],
) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(b"".join(record["packet"] for record in records))
    os.replace(temporary, path)


def configure_denoiser(
    model: YodaHybridCodec,
    generator: ConditionalDriftGenerator | None,
) -> None:
    if generator is None:
        model.denoiser_name = "original"
        return
    model.denoiser_name = "custom"
    model.denoiser = generator


def encode_mode(
    model: YodaHybridCodec,
    frames: np.ndarray,
    intra_cache: dict[str, Any],
    args: argparse.Namespace,
    device: torch.device,
    mode: str,
    output_dir: Path,
    collect_training: bool = False,
) -> dict[str, Any]:
    from src.rt_modules.utils.stream_helper import SPSHelper

    bin_path = output_dir / f"artifacts/{mode}.bin"
    manifest_path = output_dir / f"artifacts/{mode}_encode.json"
    training_path = output_dir / "artifacts/training_cache.pt"
    if (
        args.reuse
        and bin_path.is_file()
        and manifest_path.is_file()
        and (not collect_training or training_path.is_file())
    ):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        return manifest

    height, width = frames.shape[1:3]
    pad_right, pad_bottom = padding(frames[0])
    sps_helper = SPSHelper()
    output = io.BytesIO()
    records: list[dict[str, Any]] = []
    training_records: list[dict[str, Any]] = []
    last_recon: torch.Tensor | None = None
    model.codec.clear_dpb()
    model.codec.set_curr_poc(0)
    model.eval()
    model.denoiser.eval()
    model.vae.eval()
    model.codec.eval()

    with torch.inference_mode():
        for frame_index, frame in enumerate(frames):
            x_padded = bgr_to_model_input(
                frame,
                device,
                pad_right,
                pad_bottom,
            )
            is_i_frame = frame_index in intra_cache["records"]
            if is_i_frame:
                intra = intra_cache["records"][frame_index]
                model.codec.clear_dpb()
                model.codec.add_ref_frame(
                    None,
                    None,
                    intra["x_denoised"].to(device),
                )
                last_recon = intra["x_hat"].to(device).clamp(-1, 1)
                bit_stream = intra["bit_stream"]
                sps = dict(intra["sps"])
                encode_seconds = float(intra["encode_seconds"])
            else:
                if last_recon is None:
                    raise RuntimeError("P frame encountered before an I frame")
                context_source = last_recon
                torch.cuda.synchronize(device)
                started = time.perf_counter()
                encoded = model.compress(x_padded, last_recon)
                torch.cuda.synchronize(device)
                encode_seconds = time.perf_counter() - started
                if not torch.isfinite(encoded["x_hat"]).all():
                    raise FloatingPointError(
                        f"non-finite reconstruction in {mode} frame {frame_index}"
                    )
                last_recon = encoded["x_hat"].clamp(-1, 1)
                bit_stream = bytes(encoded["bit_stream"])
                sps = {
                    "sps_id": 0,
                    "height": height,
                    "width": width,
                    "ec_part": 0,
                    "use_ada_i": 0,
                }
                if collect_training and frame_index < args.train_frames:
                    training_records.append(
                        {
                            "frame": frame_index,
                            "condition": encoded["lq_latent_hat"][
                                0, : model.latent_channels
                            ]
                            .detach()
                            .half()
                            .cpu(),
                            # The YODA VAE expects the one-step SANA output, not
                            # the raw TA-AE latent. Treat these official denoised
                            # latents as the positive data distribution for the
                            # conditional Drifting replacement.
                            "target": encoded["x_denoised"][0, : model.latent_channels]
                            .detach()
                            .half()
                            .cpu(),
                            "target_source": "yoda_original_sana_x_denoised",
                            "last_recon": context_source[0]
                            .detach()
                            .half()
                            .cpu(),
                        }
                    )

            packet = io.BytesIO()
            packet_bytes = add_sps_and_packet(
                packet,
                sps_helper,
                bit_stream,
                is_i_frame=is_i_frame,
                sps=sps,
            )
            packet_value = packet.getvalue()
            records.append(
                {
                    "frame": frame_index,
                    "is_i_frame": is_i_frame,
                    "bit_bytes": packet_bytes,
                    "encode_seconds": encode_seconds,
                    "packet": packet_value,
                }
            )
            print(
                f"encode mode={mode:14s} frame={frame_index:03d} "
                f"{'I' if is_i_frame else 'P'} {packet_bytes * 8:8d} bits "
                f"{encode_seconds * 1000:8.2f} ms"
            )

    write_bitstream_file(bin_path, records)
    manifest = {
        "mode": mode,
        "frames": len(records),
        "height": height,
        "width": width,
        "bin": str(bin_path),
        "bin_bytes": bin_path.stat().st_size,
        "records": [
            {key: value for key, value in record.items() if key != "packet"}
            for record in records
        ],
    }
    json_dump(manifest_path, manifest)
    if collect_training:
        torch.save(training_records, training_path)
    model.codec.clear_dpb()
    torch.cuda.empty_cache()
    return manifest


def train_generator(
    model: YodaHybridCodec,
    training_cache_path: Path,
    generator_path: Path,
    history_path: Path,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[ConditionalDriftGenerator, dict[str, Any]]:
    if args.reuse and generator_path.is_file() and history_path.is_file():
        history = json.loads(history_path.read_text(encoding="utf-8"))
        generator = ConditionalDriftGenerator(
            latent_channels=32,
            context_channels=(1024, 512, 512, 256, 128),
            hidden_channels=args.hidden_channels,
            depth=args.depth,
            noise_scale=args.noise_scale,
        ).to(device)
        generator.load_state_dict(
            torch.load(generator_path, map_location=device, weights_only=True)
        )
        generator.eval()
        return generator, history

    cache = torch.load(training_cache_path, map_location="cpu", weights_only=False)
    if not cache:
        raise RuntimeError("no P-frame training records were collected")
    target_shape = tuple(cache[0]["target"].shape)
    if len(target_shape) < 3:
        raise RuntimeError(
            "training target must contain channel, height and width dimensions, "
            f"got {target_shape}"
        )
    # A cached item may be (C,H,W) or include leading singleton frame axes.
    # The generator always consumes the final three dimensions as the latent.
    latent_shape = target_shape[-3:]
    generator = ConditionalDriftGenerator(
        latent_channels=int(latent_shape[0]),
        context_channels=(1024, 512, 512, 256, 128),
        hidden_channels=args.hidden_channels,
        depth=args.depth,
        noise_scale=args.noise_scale,
    ).to(device)
    bucket_count = max(int(item["frame"]) for item in cache) + 1
    configure_denoiser(model, generator)
    generator.train()
    model.vae.eval()
    optimizer = torch.optim.AdamW(generator.parameters(), lr=args.lr, weight_decay=1e-4)
    trainer = ConditionalDriftTrainer(
        generator,
        latent_shape=latent_shape,
        num_buckets=bucket_count,
        max_positive=args.max_positive,
        max_negative=args.max_negative,
        positive_weight=1.0,
        negative_weight=1.0,
        reconstruction_weight=args.reconstruction_weight,
        R_list=(0.2, 0.05, 0.02),
        bank_device="cpu",
    )
    generator_cpu = torch.Generator().manual_seed(args.seed)
    history: dict[str, Any] = {
        "records": len(cache),
        "steps": args.train_steps,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "reconstruction_weight": args.reconstruction_weight,
        "max_positive": args.max_positive,
        "max_negative": args.max_negative,
        "grad_clip": args.grad_clip,
        "noise_scale": args.noise_scale,
        "hidden_channels": args.hidden_channels,
        "depth": args.depth,
        "condition_buckets": "frame_index",
        "latent_shape": list(latent_shape),
        "positive_source": cache[0].get("target_source", "unknown"),
        "updates": [],
    }
    started = time.perf_counter()
    for step in range(args.train_steps):
        indices = torch.randint(
            len(cache),
            (min(args.batch_size, len(cache)),),
            generator=generator_cpu,
        ).tolist()
        conditions = torch.stack(
            [cache[index]["condition"].float() for index in indices]
        ).to(device)
        targets = torch.stack(
            [cache[index]["target"].float() for index in indices]
        ).to(device)
        context_sources = torch.stack(
            [cache[index]["last_recon"].float() for index in indices]
        ).to(device)
        if conditions.ndim == 5 and conditions.shape[1] == 1:
            conditions = conditions[:, 0]
        if targets.ndim == 5 and targets.shape[1] == 1:
            targets = targets[:, 0]
        if conditions.shape[1] > model.latent_channels:
            conditions = conditions[:, : model.latent_channels]
        if conditions.shape != targets.shape:
            raise RuntimeError(
                "training conditions and targets must have matching shapes, "
                f"got {tuple(conditions.shape)} and {tuple(targets.shape)}"
            )
        if context_sources.ndim == 5 and context_sources.shape[1] == 1:
            context_sources = context_sources[:, 0]
        if context_sources.ndim != 4:
            raise RuntimeError(
                "training reconstruction context must be a 4D image batch, "
                f"got {tuple(context_sources.shape)}"
            )
        with torch.no_grad():
            contexts = model.vae.decoder.feature_extractor(context_sources)
            contexts = [value.detach() for value in contexts]
        noise = torch.randn_like(conditions)
        metrics = trainer.step(
            conditions,
            targets,
            contexts,
            optimizer,
            noise=noise,
            bucket_ids=torch.tensor(
                [int(cache[index]["frame"]) for index in indices],
                dtype=torch.long,
            ),
            max_grad_norm=args.grad_clip,
        )
        metrics["step"] = step
        metrics["frames"] = [int(cache[index]["frame"]) for index in indices]
        history["updates"].append(metrics)
        if step == 0 or (step + 1) % 25 == 0 or step + 1 == args.train_steps:
            print(
                f"train step={step + 1:04d}/{args.train_steps} "
                f"loss={metrics['loss']:.6g} drift={metrics['drift']:.6g} "
                f"recon={metrics['reconstruction']:.6g} "
                f"grad={metrics['grad_norm']:.4g}"
            )
    elapsed = time.perf_counter() - started
    generator.eval()
    torch.save(generator.state_dict(), generator_path)
    history["elapsed_seconds"] = elapsed
    history["steps_per_second"] = args.train_steps / max(elapsed, 1e-9)
    history["first_loss"] = history["updates"][0]["loss"]
    tail = min(25, len(history["updates"]))
    history["final_loss_mean"] = float(
        np.mean([item["loss"] for item in history["updates"][-tail:]])
    )
    history["final_reconstruction_mean"] = float(
        np.mean([item["reconstruction"] for item in history["updates"][-tail:]])
    )
    history["peak_cuda_memory_mb"] = torch.cuda.max_memory_allocated(device) / (
        1024**2
    )
    json_dump(history_path, history)
    return generator, history


def load_inter_model(
    args: argparse.Namespace,
    device: torch.device,
    denoiser: str,
) -> YodaHybridCodec:
    model = YodaHybridCodec.from_pretrained(
        args.sd_path,
        args.inter_checkpoint,
        prompt_embeds_path=args.prompt_embeds,
        prompt_attention_mask_path=args.prompt_attention_mask,
        denoiser=denoiser,
        upstream_root=UPSTREAM_ROOT,
        device=device,
        lora_rank=args.lora_rank,
        compute_dtype=args.precision,
    )
    model.codec.update()
    model.eval()
    model.requires_grad_(False)
    if denoiser == "original" and model.load_report.get("denoiser", 0) < 400:
        raise RuntimeError(f"inter LoRA matching is unexpectedly low: {model.load_report}")
    return model


def prepare_and_encode(
    args: argparse.Namespace,
    output_dir: Path,
    device: torch.device,
) -> tuple[ConditionalDriftGenerator, dict[str, Any]]:
    frames_path = output_dir / "artifacts/frames.npy"
    if args.reuse and frames_path.is_file():
        frames = load_frames_cache(frames_path)
        capture = cv2.VideoCapture(str(args.input))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        capture.release()
    else:
        decoded_frames, fps = load_frames(args.input, args.max_frames)
        save_frames_cache(frames_path, decoded_frames, reuse=False)
        frames = load_frames_cache(frames_path)
    if len(frames) != args.max_frames:
        raise RuntimeError(f"frame cache has {len(frames)} frames")
    if not 0 < args.train_frames < len(frames):
        raise ValueError("train-frames must leave at least one held-out frame")
    if args.intra_period <= 0:
        raise ValueError("intra-period must be positive")

    intra_cache = precompute_intra(args, frames, fps, output_dir, device)
    model = load_inter_model(args, device, "original")
    original_manifest = encode_mode(
        model,
        frames,
        intra_cache,
        args,
        device,
        "yoda_original",
        output_dir,
        collect_training=True,
    )

    identity = ConditionalDriftGenerator(
        latent_channels=32,
        context_channels=(1024, 512, 512, 256, 128),
        hidden_channels=args.hidden_channels,
        depth=args.depth,
        noise_scale=0.0,
    ).to(device)
    identity.eval()
    configure_denoiser(model, identity)
    identity_manifest = encode_mode(
        model,
        frames,
        intra_cache,
        args,
        device,
        "drift_identity",
        output_dir,
    )

    generator, history = train_generator(
        model,
        output_dir / "artifacts/training_cache.pt",
        output_dir / "artifacts/generator.pt",
        output_dir / "train_history.json",
        args,
        device,
    )
    configure_denoiser(model, generator)
    trained_manifest = encode_mode(
        model,
        frames,
        intra_cache,
        args,
        device,
        "drift_trained",
        output_dir,
    )
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return generator, {
        "yoda_original": original_manifest,
        "drift_identity": identity_manifest,
        "drift_trained": trained_manifest,
        "train_history": history,
        "intra_cache": intra_cache,
    }


def create_raw_memmap(path: Path, frames: int, height: int, width: int) -> np.memmap:
    path.parent.mkdir(parents=True, exist_ok=True)
    return np.memmap(
        path,
        mode="w+",
        dtype=np.uint8,
        shape=(frames, height, width, 3),
    )


def load_raw_memmap(
    path: Path,
    frames: int,
    height: int,
    width: int,
) -> np.memmap:
    return np.memmap(
        path,
        mode="r",
        dtype=np.uint8,
        shape=(frames, height, width, 3),
    )


def decode_mode(
    model: YodaHybridCodec,
    intra_cache: dict[str, Any],
    args: argparse.Namespace,
    device: torch.device,
    mode: str,
    output_dir: Path,
) -> dict[str, Any]:
    from src.rt_modules.utils.stream_helper import (
        NalType,
        SPSHelper,
        read_header,
        read_ip_remaining,
        read_sps_remaining,
    )

    bin_path = output_dir / f"artifacts/{mode}.bin"
    video_path = output_dir / f"{mode}.mp4"
    raw_path = output_dir / f"artifacts/{mode}.rgb"
    timing_path = output_dir / f"artifacts/{mode}_decode.json"
    done_path = output_dir / f"artifacts/{mode}.decode.done"
    if args.reuse and done_path.is_file() and timing_path.is_file():
        return json.loads(timing_path.read_text(encoding="utf-8"))

    encode_manifest = json.loads(
        (output_dir / f"artifacts/{mode}_encode.json").read_text(encoding="utf-8")
    )
    frame_count = int(encode_manifest["frames"])
    height = int(encode_manifest["height"])
    width = int(encode_manifest["width"])
    raw = create_raw_memmap(raw_path, frame_count, height, width)
    video_tmp = video_path.with_suffix(".tmp.mp4")
    writer = FFmpegWriter(video_tmp, width, height, float(intra_cache["fps"]))
    sps_helper = SPSHelper()
    last_recon: torch.Tensor | None = None
    timings: list[dict[str, Any]] = []
    model.codec.clear_dpb()
    model.codec.set_curr_poc(0)
    model.eval()
    model.denoiser.eval()
    model.codec.eval()
    model.vae.eval()

    try:
        with torch.inference_mode(), bin_path.open("rb") as input_file:
            input_buffer = io.BytesIO(input_file.read())
            for frame_index in range(frame_count):
                frame_started = time.perf_counter()
                header = read_header(input_buffer)
                while header["nal_type"] == NalType.NAL_SPS:
                    sps = read_sps_remaining(input_buffer, header["sps_id"])
                    sps_helper.add_sps_by_id(sps)
                    header = read_header(input_buffer)
                sps = sps_helper.get_sps_by_id(header["sps_id"])
                _, bit_stream = read_ip_remaining(input_buffer)
                if header["nal_type"] == NalType.NAL_I:
                    intra = intra_cache["records"][frame_index]
                    if bit_stream != intra["bit_stream"]:
                        raise RuntimeError(f"I-frame bitstream mismatch at {frame_index}")
                    model.codec.clear_dpb()
                    model.codec.add_ref_frame(
                        None,
                        None,
                        intra["x_denoised"].to(device),
                    )
                    image = intra["x_hat"].to(device)
                    last_recon = image.clamp(-1, 1)
                    compute_seconds = float(intra["decode_seconds"])
                elif header["nal_type"] == NalType.NAL_P:
                    if last_recon is None:
                        raise RuntimeError("P frame decoded before an I frame")
                    torch.cuda.synchronize(device)
                    compute_started = time.perf_counter()
                    decoded = model.decompress(bit_stream, sps, last_recon)
                    torch.cuda.synchronize(device)
                    compute_seconds = time.perf_counter() - compute_started
                    image = decoded["x_hat"]
                    if not torch.isfinite(image).all():
                        raise FloatingPointError(
                            f"non-finite decoded output in {mode} frame {frame_index}"
                        )
                    last_recon = image.clamp(-1, 1)
                else:
                    raise RuntimeError(f"unexpected NAL type: {header['nal_type']}")

                bgr = tensor_to_bgr(image, height, width)
                writer.write(bgr)
                raw[frame_index] = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                total_seconds = time.perf_counter() - frame_started
                timings.append(
                    {
                        "frame": frame_index,
                        "is_i_frame": header["nal_type"] == NalType.NAL_I,
                        "decode_seconds": total_seconds,
                        "compute_seconds": compute_seconds,
                    }
                )
                print(
                    f"decode mode={mode:14s} frame={frame_index:03d} "
                    f"{'I' if header['nal_type'] == NalType.NAL_I else 'P'} "
                    f"{total_seconds * 1000:8.2f} ms"
                )
            if input_buffer.read(1):
                raise RuntimeError(f"trailing bytes in {bin_path}")
        raw.flush()
        writer.close()
        os.replace(video_tmp, video_path)
        manifest = {
            "mode": mode,
            "frames": frame_count,
            "timings": timings,
            "video": str(video_path),
            "raw": str(raw_path),
        }
        json_dump(timing_path, manifest)
        done_path.write_text("ok\n", encoding="ascii")
        return manifest
    finally:
        if not writer.closed:
            writer.close()
        del raw


def decode_all_modes(
    args: argparse.Namespace,
    output_dir: Path,
    device: torch.device,
    intra_cache: dict[str, Any],
) -> dict[str, Any]:
    model = load_inter_model(args, device, "original")
    results: dict[str, Any] = {}
    configure_denoiser(model, None)
    results["yoda_original"] = decode_mode(
        model,
        intra_cache,
        args,
        device,
        "yoda_original",
        output_dir,
    )

    identity = ConditionalDriftGenerator(
        latent_channels=32,
        context_channels=(1024, 512, 512, 256, 128),
        hidden_channels=args.hidden_channels,
        depth=args.depth,
        noise_scale=0.0,
    ).to(device)
    identity.eval()
    configure_denoiser(model, identity)
    results["drift_identity"] = decode_mode(
        model,
        intra_cache,
        args,
        device,
        "drift_identity",
        output_dir,
    )

    trained = ConditionalDriftGenerator(
        latent_channels=32,
        context_channels=(1024, 512, 512, 256, 128),
        hidden_channels=args.hidden_channels,
        depth=args.depth,
        noise_scale=args.noise_scale,
    ).to(device)
    trained.load_state_dict(
        torch.load(
            output_dir / "artifacts/generator.pt",
            map_location=device,
            weights_only=True,
        )
    )
    trained.eval()
    configure_denoiser(model, trained)
    results["drift_trained"] = decode_mode(
        model,
        intra_cache,
        args,
        device,
        "drift_trained",
        output_dir,
    )
    del model, identity, trained
    gc.collect()
    torch.cuda.empty_cache()
    return results


def psnr_per_image(reference: torch.Tensor, reconstructed: torch.Tensor) -> torch.Tensor:
    mse = (reference - reconstructed).square().flatten(1).mean(dim=1)
    return -10.0 * torch.log10(mse.clamp_min(1e-12))


def load_metric_suite(device: torch.device) -> dict[str, Any]:
    import lpips
    import piq
    from pytorch_msssim import ms_ssim

    perceptual = lpips.LPIPS(net="alex", verbose=False).to(device).eval()
    dists = piq.DISTS(reduction="none").to(device).eval()
    return {
        "lpips": perceptual,
        "dists": dists,
        "ms_ssim": ms_ssim,
    }


def compute_metrics(
    args: argparse.Namespace,
    output_dir: Path,
    device: torch.device,
) -> dict[str, Any]:
    frames = load_frames_cache(output_dir / "artifacts/frames.npy")
    frame_count, height, width = frames.shape[:3]
    metrics_path = output_dir / "per_frame_metrics.json"
    suite = load_metric_suite(device)
    modes = list(MODE_LABELS)
    per_mode_values: dict[str, list[dict[str, float]]] = {mode: [] for mode in modes}
    batch_size = 8

    for mode in modes:
        encode_manifest = json.loads(
            (output_dir / f"artifacts/{mode}_encode.json").read_text(encoding="utf-8")
        )
        decode_manifest = json.loads(
            (output_dir / f"artifacts/{mode}_decode.json").read_text(encoding="utf-8")
        )
        raw = load_raw_memmap(
            output_dir / f"artifacts/{mode}.rgb",
            frame_count,
            height,
            width,
        )
        values: list[dict[str, float]] = []
        with torch.inference_mode():
            for start in range(0, frame_count, batch_size):
                end = min(start + batch_size, frame_count)
                reference_np = np.stack(
                    [
                        cv2.cvtColor(frames[index], cv2.COLOR_BGR2RGB)
                        for index in range(start, end)
                    ]
                )
                reconstructed_np = np.asarray(raw[start:end])
                reference = (
                    torch.from_numpy(reference_np)
                    .to(device)
                    .permute(0, 3, 1, 2)
                    .float()
                    / 255.0
                )
                reconstructed = (
                    torch.from_numpy(reconstructed_np)
                    .to(device)
                    .permute(0, 3, 1, 2)
                    .float()
                    / 255.0
                )
                psnr = psnr_per_image(reference, reconstructed)
                msssim = suite["ms_ssim"](
                    reference,
                    reconstructed,
                    data_range=1.0,
                    size_average=False,
                )
                lpips_value = suite["lpips"](reference * 2 - 1, reconstructed * 2 - 1)
                lpips_value = lpips_value.reshape(-1)
                dists_value = suite["dists"](reference, reconstructed).reshape(-1)
                for offset, index in enumerate(range(start, end)):
                    encode_record = encode_manifest["records"][index]
                    decode_record = decode_manifest["timings"][index]
                    bpp = encode_record["bit_bytes"] * 8 / (height * width)
                    values.append(
                        {
                            "psnr": float(psnr[offset].cpu()),
                            "msssim": float(msssim[offset].cpu()),
                            "lpips": float(lpips_value[offset].cpu()),
                            "dists": float(dists_value[offset].cpu()),
                            "bpp": bpp,
                            "bit_bytes": int(encode_record["bit_bytes"]),
                            "is_i_frame": bool(encode_record["is_i_frame"]),
                            "encode_ms": float(encode_record["encode_seconds"] * 1000),
                            "decode_ms": float(
                                decode_record["decode_seconds"] * 1000
                            ),
                        }
                    )
        per_mode_values[mode] = values
        mean = {
            key: float(np.mean([item[key] for item in values]))
            for key in ("psnr", "msssim", "lpips", "dists", "bpp", "encode_ms", "decode_ms")
        }
        print(f"metrics mode={mode:14s} " + " ".join(
            f"{key}={value:.6g}" for key, value in mean.items()
        ))

    frame_entries = []
    for index in range(frame_count):
        entry = {
            "frame": index,
            "split": "train" if index < args.train_frames else "eval",
            "modes": {
                mode: per_mode_values[mode][index]
                for mode in modes
            },
        }
        frame_entries.append(entry)
    json_dump(metrics_path, {"frames": frame_entries})

    del suite
    gc.collect()
    torch.cuda.empty_cache()
    return {
        "per_mode": per_mode_values,
        "frame_entries": frame_entries,
    }


def aggregate(values: Iterable[dict[str, float]]) -> dict[str, Any]:
    values = list(values)
    result: dict[str, Any] = {"frames": len(values)}
    for key in ("psnr", "msssim", "lpips", "dists", "bpp", "encode_ms", "decode_ms"):
        array = np.asarray([item[key] for item in values], dtype=np.float64)
        result[key] = {
            "mean": float(array.mean()),
            "std": float(array.std()),
            "min": float(array.min()),
            "max": float(array.max()),
        }
    return result


def draw_label(frame: np.ndarray, label: str) -> np.ndarray:
    output = frame.copy()
    cv2.rectangle(output, (0, 0), (min(output.shape[1], 155), 22), (0, 0, 0), -1)
    cv2.putText(
        output,
        label,
        (5, 16),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return output


def render_videos(
    args: argparse.Namespace,
    output_dir: Path,
    intra_cache: dict[str, Any],
) -> None:
    frames = load_frames_cache(output_dir / "artifacts/frames.npy")
    frame_count, height, width = frames.shape[:3]
    fps = float(intra_cache["fps"])
    raw = {
        mode: load_raw_memmap(
            output_dir / f"artifacts/{mode}.rgb",
            frame_count,
            height,
            width,
        )
        for mode in MODE_LABELS
    }

    original_writer = FFmpegWriter(output_dir / "original.mp4", width, height, fps)
    side_writer = FFmpegWriter(
        output_dir / "comparison_side_by_side.mp4",
        width * 4,
        height,
        fps,
    )
    grid_writer = FFmpegWriter(
        output_dir / "comparison_2x2.mp4",
        width * 2,
        height * 2,
        fps,
    )
    yoda_error_writer = FFmpegWriter(
        output_dir / "error_yoda.mp4",
        width,
        height,
        fps,
    )
    drift_error_writer = FFmpegWriter(
        output_dir / "error_drift.mp4",
        width,
        height,
        fps,
    )
    try:
        for index in range(frame_count):
            original_bgr = np.asarray(frames[index])
            mode_bgr = {
                mode: cv2.cvtColor(np.asarray(raw[mode][index]), cv2.COLOR_RGB2BGR)
                for mode in MODE_LABELS
            }
            original_writer.write(original_bgr)
            side = np.concatenate(
                [
                    draw_label(original_bgr, "Original"),
                    draw_label(mode_bgr["yoda_original"], "YODA"),
                    draw_label(mode_bgr["drift_identity"], "Drift identity"),
                    draw_label(mode_bgr["drift_trained"], "Drift trained"),
                ],
                axis=1,
            )
            grid = np.concatenate(
                [
                    np.concatenate(
                        [
                            draw_label(original_bgr, "Original"),
                            draw_label(mode_bgr["yoda_original"], "YODA"),
                        ],
                        axis=1,
                    ),
                    np.concatenate(
                        [
                            draw_label(mode_bgr["drift_identity"], "Drift identity"),
                            draw_label(mode_bgr["drift_trained"], "Drift trained"),
                        ],
                        axis=1,
                    ),
                ],
                axis=0,
            )
            side_writer.write(side)
            grid_writer.write(grid)

            for mode, writer, label in (
                ("yoda_original", yoda_error_writer, "Error YODA x4"),
                ("drift_trained", drift_error_writer, "Error Drift x4"),
            ):
                difference = np.abs(
                    original_bgr.astype(np.int16)
                    - mode_bgr[mode].astype(np.int16)
                ).astype(np.uint8)
                difference = np.clip(difference.astype(np.uint16) * 4, 0, 255).astype(
                    np.uint8
                )
                error_map = cv2.applyColorMap(
                    cv2.cvtColor(difference, cv2.COLOR_BGR2GRAY),
                    cv2.COLORMAP_INFERNO,
                )
                writer.write(draw_label(error_map, label))
    finally:
        original_writer.close()
        side_writer.close()
        grid_writer.close()
        yoda_error_writer.close()
        drift_error_writer.close()


def probe_video(path: Path) -> dict[str, Any]:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-count_frames",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=codec_name,width,height,pix_fmt,r_frame_rate,avg_frame_rate,nb_frames,nb_read_frames",
        "-show_entries",
        "format=duration,size",
        "-of",
        "json",
        str(path),
    ]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    return json.loads(result.stdout)


def validate_video(path: Path, expected_frames: int, width: int, height: int) -> dict[str, Any]:
    probe = probe_video(path)
    stream = probe["streams"][0]
    codec = stream.get("codec_name")
    pixels = stream.get("pix_fmt")
    read_frames = int(stream.get("nb_read_frames", stream.get("nb_frames", 0)))
    if codec != "h264":
        raise RuntimeError(f"{path} codec is {codec}, expected h264")
    if pixels != "yuv420p":
        raise RuntimeError(f"{path} pixel format is {pixels}, expected yuv420p")
    if (int(stream["width"]), int(stream["height"])) != (width, height):
        raise RuntimeError(f"{path} has unexpected dimensions")
    if read_frames != expected_frames:
        raise RuntimeError(f"{path} has {read_frames} frames, expected {expected_frames}")

    capture = cv2.VideoCapture(str(path))
    sampled_means: list[float] = []
    sampled_stds: list[float] = []
    decoded = 0
    sampled_frames: list[np.ndarray] = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if decoded % max(expected_frames // 30, 1) == 0:
            sampled_means.append(float(frame.mean()))
            sampled_stds.append(float(frame.std()))
            if len(sampled_frames) < 12:
                sampled_frames.append(frame.astype(np.float32))
        decoded += 1
    capture.release()
    if decoded != expected_frames:
        raise RuntimeError(f"OpenCV decoded {decoded} frames from {path}")
    if min(sampled_stds) < 2.0 or max(sampled_means) < 5.0:
        raise RuntimeError(f"{path} appears blank or flat")
    return {
        "path": str(path),
        "codec": codec,
        "pix_fmt": pixels,
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "frames": read_frames,
        "avg_frame_rate": stream.get("avg_frame_rate"),
        "duration": float(probe["format"]["duration"]),
        "sampled_mean_min": min(sampled_means),
        "sampled_mean_max": max(sampled_means),
        "sampled_std_min": min(sampled_stds),
        "sampled_frames": sampled_frames,
    }


def validate_all_videos(
    output_dir: Path,
    expected_frames: int,
    width: int,
    height: int,
) -> dict[str, Any]:
    dimensions = {
        "original.mp4": (width, height),
        "yoda_original.mp4": (width, height),
        "drift_identity.mp4": (width, height),
        "drift_trained.mp4": (width, height),
        "comparison_side_by_side.mp4": (width * 4, height),
        "comparison_2x2.mp4": (width * 2, height * 2),
        "error_yoda.mp4": (width, height),
        "error_drift.mp4": (width, height),
    }
    report = {
        name: validate_video(
            output_dir / name,
            expected_frames,
            expected_width,
            expected_height,
        )
        for name, (expected_width, expected_height) in dimensions.items()
    }
    reference = report["original.mp4"]["sampled_frames"]
    reconstructed = report["yoda_original.mp4"]["sampled_frames"]
    drift = report["drift_trained.mp4"]["sampled_frames"]
    yoda_drift_delta = float(
        np.mean(
            [
                np.abs(a - b).mean()
                for a, b in zip(reconstructed, drift)
            ]
        )
    )
    original_yoda_delta = float(
        np.mean(
            [
                np.abs(a - b).mean()
                for a, b in zip(reference, reconstructed)
            ]
        )
    )
    if yoda_drift_delta < 0.1:
        raise RuntimeError("YODA and trained Drifting videos are unexpectedly identical")
    if original_yoda_delta < 0.1:
        raise RuntimeError("YODA reconstruction is unexpectedly identical to source")
    for value in report.values():
        value.pop("sampled_frames", None)
    report["cross_video_difference"] = {
        "original_vs_yoda_mean_abs": original_yoda_delta,
        "yoda_vs_drift_trained_mean_abs": yoda_drift_delta,
    }
    return report


def finalize_outputs(
    args: argparse.Namespace,
    output_dir: Path,
    device: torch.device,
    checkpoint_report: dict[str, Any],
    encode_report: dict[str, Any],
    decode_report: dict[str, Any],
    metric_report: dict[str, Any],
) -> dict[str, Any]:
    render_videos(args, output_dir, encode_report["intra_cache"])
    frames = load_frames_cache(output_dir / "artifacts/frames.npy")
    frame_count, height, width = frames.shape[:3]
    validation = validate_all_videos(output_dir, frame_count, width, height)
    json_dump(output_dir / "validation.json", validation)

    eval_start = args.train_frames
    summaries: dict[str, Any] = {}
    for mode in MODE_LABELS:
        values = metric_report["per_mode"][mode]
        summaries[mode] = {
            "all_frames": aggregate(values),
            "held_out": aggregate(values[eval_start:]),
            "training_segment": aggregate(values[:eval_start]),
        }
    yoda_eval = summaries["yoda_original"]["held_out"]
    drift_eval = summaries["drift_trained"]["held_out"]
    deltas = {
        key: drift_eval[key]["mean"] - yoda_eval[key]["mean"]
        for key in ("psnr", "msssim", "lpips", "dists", "bpp", "encode_ms", "decode_ms")
    }
    train_history = encode_report["train_history"]
    summary = {
        "status": "ok",
        "scope": (
            "Akiyo single-video subset: YODA reproduction and conditional "
            "Drifting replacement experiment"
        ),
        "claim": (
            "This is not a full benchmark or a claim of improvement over YODA. "
            "Only frames after the training split are used for held-out metrics."
        ),
        "papers": {
            "yoda": "arXiv:2601.01141v1",
            "drifting": "arXiv:2602.04770v2",
        },
        "dataset": {
            "input": str(args.input),
            "frames": frame_count,
            "train_frames": args.train_frames,
            "held_out_frames": frame_count - args.train_frames,
            "height": height,
            "width": width,
            "intra_period": args.intra_period,
        },
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device": str(device),
            "gpu": torch.cuda.get_device_name(device),
        },
        "checkpoints": checkpoint_report,
        "model_load_report": {
            "intra": encode_report["intra_cache"]["load_report"],
        },
        "training": {
            "steps": train_history["steps"],
            "batch_size": train_history["batch_size"],
            "records": train_history["records"],
            "elapsed_seconds": train_history["elapsed_seconds"],
            "first_loss": train_history["first_loss"],
            "final_loss_mean": train_history["final_loss_mean"],
            "final_reconstruction_mean": train_history[
                "final_reconstruction_mean"
            ],
            "reconstruction_weight": train_history["reconstruction_weight"],
            "grad_clip": train_history["grad_clip"],
            "noise_scale": train_history["noise_scale"],
            "condition_buckets": train_history["condition_buckets"],
        },
        "metrics": summaries,
        "drift_trained_minus_yoda_held_out": deltas,
        "bitstreams": {
            mode: {
                "path": str(output_dir / f"artifacts/{mode}.bin"),
                "bytes": (output_dir / f"artifacts/{mode}.bin").stat().st_size,
            }
            for mode in MODE_LABELS
        },
        "videos": {
            "original": str(output_dir / "original.mp4"),
            "yoda_original": str(output_dir / "yoda_original.mp4"),
            "drift_identity": str(output_dir / "drift_identity.mp4"),
            "drift_trained": str(output_dir / "drift_trained.mp4"),
            "comparison_side_by_side": str(
                output_dir / "comparison_side_by_side.mp4"
            ),
            "comparison_2x2": str(output_dir / "comparison_2x2.mp4"),
            "error_yoda": str(output_dir / "error_yoda.mp4"),
            "error_drift": str(output_dir / "error_drift.mp4"),
        },
        "validation": validation["cross_video_difference"],
        "decode": decode_report,
    }
    json_dump(output_dir / "summary.json", summary)
    return summary


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "artifacts").mkdir(parents=True, exist_ok=True)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the real YODA experiment")
    device = torch.device(args.device)
    if device.type != "cuda":
        raise ValueError("use a CUDA device for this experiment")
    if args.batch_size <= 0 or args.train_steps <= 0:
        raise ValueError("batch-size and train-steps must be positive")
    set_seed(args.seed)

    checkpoint_report = verify_checkpoints(args)
    if args.hash_cache_only:
        json_dump(output_dir / "checkpoint_hashes.json", checkpoint_report)
        return

    _, encode_report = prepare_and_encode(args, output_dir, device)
    decode_report = decode_all_modes(
        args,
        output_dir,
        device,
        encode_report["intra_cache"],
    )
    metric_report = compute_metrics(args, output_dir, device)
    summary = finalize_outputs(
        args,
        output_dir,
        device,
        checkpoint_report,
        encode_report,
        decode_report,
        metric_report,
    )
    print(json.dumps({
        "status": summary["status"],
        "held_out": {
            mode: summary["metrics"][mode]["held_out"]
            for mode in MODE_LABELS
        },
        "videos": summary["videos"],
    }, indent=2))


if __name__ == "__main__":
    main()
