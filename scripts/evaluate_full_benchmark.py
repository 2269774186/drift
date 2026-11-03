#!/usr/bin/env python3
"""Evaluate YODA and the trained Drifting replacement on full YUV videos.

The benchmark reads raw 8-bit I420 frames with BT.709 conversion, preserves
YODA's real TA-AE/CLC bitstream path, and writes complete H.264 comparisons.
It is intentionally sequential so temporary 1080p RGB buffers can be removed
after each video.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import platform
import re
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable, Iterator

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SCRIPTS_ROOT = ROOT / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from run_real_yoda_drift import (  # noqa: E402
    FFmpegWriter,
    configure_denoiser,
    decode_mode,
    encode_mode,
    json_dump,
    load_inter_model,
    load_metric_suite,
    precompute_intra,
    psnr_per_image,
    validate_video,
)
from yoda_drift.drift_model import ConditionalDriftGenerator  # noqa: E402


GROUP_LABELS = {
    "uvg": "UVG",
    "hevc_b": "HEVC-B",
    "mcl_jcv": "MCL-JCV",
}
METRIC_KEYS = (
    "psnr",
    "msssim",
    "lpips",
    "dists",
    "bpp",
    "encode_ms",
    "decode_ms",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-root", type=Path, default=Path("/data1/zy/test"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/data1/zy/yoda_drift_full/eval"),
    )
    parser.add_argument(
        "--generator",
        type=Path,
        default=Path("/data1/zy/yoda_drift_full/train/generator.pt"),
    )
    parser.add_argument(
        "--training-history",
        type=Path,
        default=Path("/data1/zy/yoda_drift_full/train/train_history.json"),
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
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--lora-rank", type=int, default=72)
    parser.add_argument("--intra-period", type=int, default=32)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--frames", type=int, default=96)
    parser.add_argument(
        "--yuv-range",
        choices=("limited", "full"),
        default="limited",
    )
    parser.add_argument(
        "--groups",
        default="uvg,hevc_b,mcl_jcv",
        help="comma-separated directory names under test-root",
    )
    parser.add_argument("--max-videos", type=int, default=0)
    parser.add_argument("--comparison-scale", type=float, default=0.5)
    parser.add_argument("--metric-batch", type=int, default=1)
    parser.add_argument("--hidden-channels", type=int, default=64)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--reuse", action="store_true")
    parser.add_argument("--keep-raw", action="store_true")
    parser.add_argument("--allow-incomplete-training", action="store_true")
    args = parser.parse_args()
    if args.width <= 0 or args.height <= 0 or args.frames <= 0:
        raise ValueError("width, height and frames must be positive")
    if args.width % 2 or args.height % 2:
        raise ValueError("YUV420 dimensions must be even")
    if args.intra_period <= 0 or args.metric_batch <= 0:
        raise ValueError("intra-period and metric-batch must be positive")
    if not 0 < args.comparison_scale <= 1:
        raise ValueError("comparison-scale must be in (0, 1]")
    args.output_dir = args.output_dir.resolve()
    return args


class YUV420Video:
    """Read one raw 8-bit I420 sequence lazily from a memory map."""

    def __init__(
        self,
        path: Path,
        width: int,
        height: int,
        frames: int,
        yuv_range: str,
    ) -> None:
        self.path = path
        self.width = int(width)
        self.height = int(height)
        self.frames = int(frames)
        self.yuv_range = yuv_range
        self.shape = (self.frames, self.height, self.width, 3)
        expected_size = self.frames * self.width * self.height * 3 // 2
        if not path.is_file():
            raise FileNotFoundError(path)
        actual_size = path.stat().st_size
        if actual_size != expected_size:
            raise ValueError(
                f"{path} has {actual_size} bytes, expected {expected_size}"
            )
        self._mapped = np.memmap(
            path,
            mode="r",
            dtype=np.uint8,
            shape=(self.frames, self.height * 3 // 2, self.width),
        )

    def __len__(self) -> int:
        return self.frames

    def __getitem__(self, index: int) -> np.ndarray:
        if index < 0:
            index += self.frames
        if not 0 <= index < self.frames:
            raise IndexError(index)
        packed = np.asarray(self._mapped[index])
        y = packed[: self.height].astype(np.float32)
        u = packed[
            self.height : self.height + self.height // 4
        ].reshape(self.height // 2, self.width // 2)
        v = packed[
            self.height + self.height // 4 :
        ].reshape(self.height // 2, self.width // 2)
        u = cv2.resize(
            u,
            (self.width, self.height),
            interpolation=cv2.INTER_NEAREST,
        ).astype(np.float32)
        v = cv2.resize(
            v,
            (self.width, self.height),
            interpolation=cv2.INTER_NEAREST,
        ).astype(np.float32)
        if self.yuv_range == "limited":
            y = (y - 16.0) / 219.0
            u = (u - 128.0) / 224.0
            v = (v - 128.0) / 224.0
        else:
            y = y / 255.0
            u = (u - 128.0) / 255.0
            v = (v - 128.0) / 255.0
        red = y + 1.5748 * v
        green = y - 0.1873 * u - 0.4681 * v
        blue = y + 1.8556 * u
        rgb = np.stack((red, green, blue), axis=-1)
        rgb = np.clip(rgb * 255.0, 0.0, 255.0).round().astype(np.uint8)
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    def __iter__(self) -> Iterator[np.ndarray]:
        for index in range(self.frames):
            yield self[index]


def discover_videos(
    test_root: Path,
    groups: Iterable[str],
    max_videos: int,
) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for group in groups:
        group_dir = test_root / group
        if not group_dir.is_dir():
            raise FileNotFoundError(f"missing benchmark group: {group_dir}")
        videos = sorted(group_dir.glob("*.yuv"))
        if not videos:
            raise RuntimeError(f"no YUV files found in {group_dir}")
        for path in videos:
            entries.append(
                {
                    "group": group,
                    "group_label": GROUP_LABELS.get(group, group),
                    "name": path.stem,
                    "path": path,
                }
            )
    if max_videos:
        entries = entries[:max_videos]
    if not entries:
        raise RuntimeError("benchmark selection is empty")
    return entries


def parse_fps(path: Path, default: float = 30.0) -> float:
    match = re.search(r"_(\d+)(?:fps)?_96f$", path.stem)
    if not match:
        return default
    value = float(match.group(1))
    if value <= 0:
        raise ValueError(f"invalid frame rate encoded in {path.name}")
    return value


def set_seed(seed: int) -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":16:8")
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


def verify_training(
    history_path: Path,
    generator_path: Path,
    allow_incomplete: bool,
) -> dict[str, Any]:
    if not history_path.is_file():
        raise FileNotFoundError(history_path)
    if not generator_path.is_file():
        raise FileNotFoundError(generator_path)
    history = json.loads(history_path.read_text(encoding="utf-8"))
    if (
        not allow_incomplete
        and history.get("status") not in {"complete", "limited"}
    ):
        raise RuntimeError(
            f"training is not complete: status={history.get('status')}"
        )
    if not allow_incomplete and history.get("status") == "limited":
        raise RuntimeError("refusing to benchmark a limited smoke checkpoint")
    return history


def load_drift_generator(
    args: argparse.Namespace,
    device: torch.device,
    training_history: dict[str, Any],
) -> ConditionalDriftGenerator:
    generator = ConditionalDriftGenerator(
        latent_channels=32,
        context_channels=(1024, 512, 512, 256, 128),
        hidden_channels=int(
            training_history.get("hidden_channels", args.hidden_channels)
        ),
        depth=int(training_history.get("depth", args.depth)),
        noise_scale=0.0,
    ).to(device)
    state = torch.load(args.generator, map_location="cpu", weights_only=True)
    generator.load_state_dict(state, strict=True)
    generator.eval()
    return generator


def write_original_video(
    video: YUV420Video,
    fps: float,
    path: Path,
) -> None:
    writer = FFmpegWriter(path, video.width, video.height, fps)
    try:
        for frame in video:
            writer.write(frame)
    finally:
        writer.close()


def create_raw_memmap(
    path: Path,
    frames: int,
    height: int,
    width: int,
) -> np.memmap:
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


def compute_mode_metrics(
    video: YUV420Video,
    output_dir: Path,
    mode: str,
    suite: dict[str, Any],
    device: torch.device,
    batch_size: int,
) -> list[dict[str, float | int | bool]]:
    encode_manifest = json.loads(
        (output_dir / f"artifacts/{mode}_encode.json").read_text(
            encoding="utf-8"
        )
    )
    decode_manifest = json.loads(
        (output_dir / f"artifacts/{mode}_decode.json").read_text(
            encoding="utf-8"
        )
    )
    raw = load_raw_memmap(
        output_dir / f"artifacts/{mode}.rgb",
        video.frames,
        video.height,
        video.width,
    )
    values: list[dict[str, float | int | bool]] = []
    with torch.inference_mode():
        for start in range(0, video.frames, batch_size):
            end = min(start + batch_size, video.frames)
            reference_np = np.stack(
                [
                    cv2.cvtColor(video[index], cv2.COLOR_BGR2RGB)
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
            lpips_value = suite["lpips"](
                reference * 2 - 1,
                reconstructed * 2 - 1,
            ).reshape(-1)
            dists_value = suite["dists"](
                reference,
                reconstructed,
            ).reshape(-1)
            for offset, index in enumerate(range(start, end)):
                encode_record = encode_manifest["records"][index]
                decode_record = decode_manifest["timings"][index]
                values.append(
                    {
                        "psnr": float(psnr[offset].cpu()),
                        "msssim": float(msssim[offset].cpu()),
                        "lpips": float(lpips_value[offset].cpu()),
                        "dists": float(dists_value[offset].cpu()),
                        "bpp": encode_record["bit_bytes"]
                        * 8
                        / (video.height * video.width),
                        "bit_bytes": int(encode_record["bit_bytes"]),
                        "is_i_frame": bool(encode_record["is_i_frame"]),
                        "encode_ms": float(
                            encode_record["encode_seconds"] * 1000
                        ),
                        "decode_ms": float(
                            decode_record["decode_seconds"] * 1000
                        ),
                    }
                )
    del raw
    return values


def aggregate(values: Iterable[dict[str, Any]]) -> dict[str, Any]:
    entries = list(values)
    result: dict[str, Any] = {"frames": len(entries)}
    for key in METRIC_KEYS:
        array = np.asarray([item[key] for item in entries], dtype=np.float64)
        result[key] = {
            "mean": float(array.mean()),
            "std": float(array.std()),
            "min": float(array.min()),
            "max": float(array.max()),
        }
    return result


def draw_label(frame: np.ndarray, label: str) -> np.ndarray:
    output = frame.copy()
    cv2.rectangle(output, (0, 0), (min(output.shape[1], 190), 30), (0, 0, 0), -1)
    cv2.putText(
        output,
        label,
        (6, 21),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return output


def render_comparison(
    video: YUV420Video,
    output_dir: Path,
    fps: float,
    scale: float,
) -> tuple[int, int]:
    panel_width = max(2, int(round(video.width * scale / 2)) * 2)
    panel_height = max(2, int(round(video.height * scale / 2)) * 2)
    raw_yoda = load_raw_memmap(
        output_dir / "artifacts/yoda_original.rgb",
        video.frames,
        video.height,
        video.width,
    )
    raw_drift = load_raw_memmap(
        output_dir / "artifacts/drift_trained.rgb",
        video.frames,
        video.height,
        video.width,
    )
    canvas_width = panel_width * 3
    writer = FFmpegWriter(
        output_dir / "comparison_side_by_side.mp4",
        canvas_width,
        panel_height,
        fps,
    )
    try:
        for index, original in enumerate(video):
            panels = [
                draw_label(
                    cv2.resize(
                        original,
                        (panel_width, panel_height),
                        interpolation=cv2.INTER_AREA,
                    ),
                    "Original",
                ),
                draw_label(
                    cv2.resize(
                        cv2.cvtColor(
                            np.asarray(raw_yoda[index]),
                            cv2.COLOR_RGB2BGR,
                        ),
                        (panel_width, panel_height),
                        interpolation=cv2.INTER_AREA,
                    ),
                    "YODA original",
                ),
                draw_label(
                    cv2.resize(
                        cv2.cvtColor(
                            np.asarray(raw_drift[index]),
                            cv2.COLOR_RGB2BGR,
                        ),
                        (panel_width, panel_height),
                        interpolation=cv2.INTER_AREA,
                    ),
                    "Drifting trained",
                ),
            ]
            writer.write(np.concatenate(panels, axis=1))
    finally:
        writer.close()
    del raw_yoda, raw_drift
    return canvas_width, panel_height


def validate_output_videos(
    video: YUV420Video,
    output_dir: Path,
    comparison_size: tuple[int, int],
) -> dict[str, Any]:
    full_size = (video.width, video.height)
    dimensions = {
        "original.mp4": full_size,
        "yoda_original.mp4": full_size,
        "drift_trained.mp4": full_size,
        "comparison_side_by_side.mp4": comparison_size,
    }
    reports: dict[str, Any] = {}
    sampled: dict[str, list[np.ndarray]] = {}
    for name, (width, height) in dimensions.items():
        report = validate_video(
            output_dir / name,
            video.frames,
            width,
            height,
        )
        sampled[name] = report.pop("sampled_frames")
        reports[name] = report
    reference = sampled["original.mp4"]
    yoda = sampled["yoda_original.mp4"]
    drift = sampled["drift_trained.mp4"]
    original_yoda_delta = float(
        np.mean(
            [
                np.abs(left - right).mean()
                for left, right in zip(reference, yoda)
            ]
        )
    )
    yoda_drift_delta = float(
        np.mean(
            [
                np.abs(left - right).mean()
                for left, right in zip(yoda, drift)
            ]
        )
    )
    if original_yoda_delta < 0.1:
        raise RuntimeError("YODA reconstruction is unexpectedly identical to source")
    if yoda_drift_delta < 0.1:
        raise RuntimeError("YODA and Drifting videos are unexpectedly identical")
    reports["cross_video_difference"] = {
        "original_vs_yoda_mean_abs": original_yoda_delta,
        "yoda_vs_drift_trained_mean_abs": yoda_drift_delta,
    }
    return reports


def cleanup_raw(output_dir: Path) -> None:
    for path in (output_dir / "artifacts").glob("*.rgb"):
        path.unlink(missing_ok=True)


def process_video(
    args: argparse.Namespace,
    entry: dict[str, Any],
    device: torch.device,
    training_history: dict[str, Any],
) -> dict[str, Any]:
    output_dir = args.output_dir / entry["group"] / entry["name"]
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "artifacts").mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.json"
    if args.reuse and summary_path.is_file():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if summary.get("status") == "ok":
            print(f"reuse group={entry['group']} video={entry['name']}")
            return summary

    started = time.perf_counter()
    fps = parse_fps(entry["path"])
    video = YUV420Video(
        entry["path"],
        args.width,
        args.height,
        args.frames,
        args.yuv_range,
    )
    local_args = SimpleNamespace(
        reuse=args.reuse,
        train_frames=args.frames,
        intra_period=args.intra_period,
        lora_rank=args.lora_rank,
        prompt_embeds=args.prompt_embeds,
        prompt_attention_mask=args.prompt_attention_mask,
        intra_checkpoint=args.intra_checkpoint,
        inter_checkpoint=args.inter_checkpoint,
        sd_path=args.sd_path,
    )
    print(
        f"start group={entry['group']} video={entry['name']} "
        f"size={video.width}x{video.height} frames={video.frames} fps={fps:g}"
    )
    original_path = output_dir / "original.mp4"
    if not (args.reuse and original_path.is_file()):
        write_original_video(video, fps, original_path)

    intra_cache = precompute_intra(
        local_args,
        video,
        fps,
        output_dir,
        device,
    )
    model = load_inter_model(local_args, device, "original")
    original_denoiser = model.denoiser
    generator = load_drift_generator(args, device, training_history)
    original_manifest = encode_mode(
        model,
        video,
        intra_cache,
        local_args,
        device,
        "yoda_original",
        output_dir,
    )
    configure_denoiser(model, generator)
    drift_manifest = encode_mode(
        model,
        video,
        intra_cache,
        local_args,
        device,
        "drift_trained",
        output_dir,
    )

    configure_denoiser(model, None)
    model.denoiser = original_denoiser
    yoda_decode = decode_mode(
        model,
        intra_cache,
        local_args,
        device,
        "yoda_original",
        output_dir,
    )
    configure_denoiser(model, generator)
    drift_decode = decode_mode(
        model,
        intra_cache,
        local_args,
        device,
        "drift_trained",
        output_dir,
    )
    del model, generator, original_denoiser
    gc.collect()
    torch.cuda.empty_cache()

    suite = load_metric_suite(device)
    per_mode = {
        "yoda_original": compute_mode_metrics(
            video,
            output_dir,
            "yoda_original",
            suite,
            device,
            args.metric_batch,
        ),
        "drift_trained": compute_mode_metrics(
            video,
            output_dir,
            "drift_trained",
            suite,
            device,
            args.metric_batch,
        ),
    }
    del suite
    gc.collect()
    torch.cuda.empty_cache()
    aggregates = {
        mode: aggregate(values)
        for mode, values in per_mode.items()
    }
    json_dump(
        output_dir / "per_frame_metrics.json",
        {"frames": per_mode},
    )
    comparison_size = render_comparison(
        video,
        output_dir,
        fps,
        args.comparison_scale,
    )
    validation = validate_output_videos(video, output_dir, comparison_size)
    json_dump(output_dir / "validation.json", validation)

    yoda_mean = aggregates["yoda_original"]
    drift_mean = aggregates["drift_trained"]
    deltas = {
        key: drift_mean[key]["mean"] - yoda_mean[key]["mean"]
        for key in METRIC_KEYS
    }
    elapsed = time.perf_counter() - started
    summary = {
        "status": "ok",
        "group": entry["group"],
        "group_label": entry["group_label"],
        "name": entry["name"],
        "source": str(entry["path"]),
        "fps": fps,
        "frames": video.frames,
        "height": video.height,
        "width": video.width,
        "yuv_range": args.yuv_range,
        "color_matrix": "BT.709",
        "intra_period": args.intra_period,
        "elapsed_seconds": elapsed,
        "metrics": aggregates,
        "drift_trained_minus_yoda": deltas,
        "bitstreams": {
            "yoda_original": {
                "path": str(
                    output_dir / "artifacts/yoda_original.bin"
                ),
                "bytes": original_manifest["bin_bytes"],
            },
            "drift_trained": {
                "path": str(
                    output_dir / "artifacts/drift_trained.bin"
                ),
                "bytes": drift_manifest["bin_bytes"],
            },
        },
        "videos": {
            "original": str(original_path),
            "yoda_original": str(output_dir / "yoda_original.mp4"),
            "drift_trained": str(output_dir / "drift_trained.mp4"),
            "comparison_side_by_side": str(
                output_dir / "comparison_side_by_side.mp4"
            ),
        },
        "validation": validation["cross_video_difference"],
        "decode": {
            "yoda_original": yoda_decode["video"],
            "drift_trained": drift_decode["video"],
        },
    }
    json_dump(summary_path, summary)
    if not args.keep_raw:
        cleanup_raw(output_dir)
    print(
        f"done group={entry['group']} video={entry['name']} "
        f"yoda_psnr={yoda_mean['psnr']['mean']:.4f} "
        f"drift_psnr={drift_mean['psnr']['mean']:.4f} "
        f"delta={deltas['psnr']:+.4f} elapsed={elapsed:.1f}s"
    )
    return summary


def make_global_summary(
    args: argparse.Namespace,
    video_summaries: list[dict[str, Any]],
    training_history: dict[str, Any],
) -> dict[str, Any]:
    per_video_values: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for summary in video_summaries:
        output_dir = Path(summary["videos"]["original"]).parent
        frame_data = json.loads(
            (output_dir / "per_frame_metrics.json").read_text(
                encoding="utf-8"
            )
        )["frames"]
        for mode, values in frame_data.items():
            per_video_values.setdefault(mode, {})[summary["name"]] = values

    overall: dict[str, Any] = {}
    group_metrics: dict[str, Any] = {}
    for mode, videos in per_video_values.items():
        combined = [
            value
            for values in videos.values()
            for value in values
        ]
        overall[mode] = aggregate(combined)
    for group, label in GROUP_LABELS.items():
        selected = [
            summary for summary in video_summaries
            if summary["group"] == group
        ]
        if not selected:
            continue
        group_metrics[group] = {"label": label, "videos": len(selected)}
        for mode in ("yoda_original", "drift_trained"):
            combined = [
                value
                for summary in selected
                for value in per_video_values[mode][summary["name"]]
            ]
            group_metrics[group][mode] = aggregate(combined)

    deltas = {
        key: overall["drift_trained"][key]["mean"]
        - overall["yoda_original"][key]["mean"]
        for key in METRIC_KEYS
    }
    summary = {
        "status": "ok",
        "papers": {
            "yoda": "arXiv:2601.01141v1",
            "drifting": "arXiv:2602.04770v2",
        },
        "scope": (
            "Full 42-video YUV benchmark: YODA reproduction and trained "
            "conditional Drifting replacement"
        ),
        "dataset": {
            "root": str(args.test_root),
            "videos": len(video_summaries),
            "groups": {
                key: sum(
                    1 for summary in video_summaries
                    if summary["group"] == key
                )
                for key in GROUP_LABELS
            },
            "frames_per_video": args.frames,
            "height": args.height,
            "width": args.width,
            "pixel_format": "yuv420p",
            "yuv_range": args.yuv_range,
            "color_matrix": "BT.709",
            "intra_period": args.intra_period,
        },
        "training": {
            "history": str(args.training_history),
            "generator": str(args.generator),
            "status": training_history.get("status"),
            "selected_clips": training_history.get("selected_clips"),
            "steps": training_history.get("steps"),
            "elapsed_seconds": training_history.get("elapsed_seconds"),
            "positive_source": training_history.get("positive_source"),
            "condition_source": training_history.get("condition_source"),
        },
        "environment": {
            "host": platform.node(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
            "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
        "metrics": overall,
        "groups": group_metrics,
        "drift_trained_minus_yoda": deltas,
        "videos": video_summaries,
    }
    json_dump(args.output_dir / "benchmark_summary.json", summary)
    json_dump(args.output_dir / "aggregate_metrics.json", {
        "metrics": overall,
        "groups": group_metrics,
        "drift_trained_minus_yoda": deltas,
    })
    return summary


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device(args.device)
    if device.type != "cuda":
        raise ValueError("use a CUDA device")
    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    training_history = verify_training(
        args.training_history,
        args.generator,
        args.allow_incomplete_training,
    )
    groups = [
        value.strip()
        for value in args.groups.split(",")
        if value.strip()
    ]
    entries = discover_videos(
        args.test_root,
        groups,
        args.max_videos,
    )
    print(
        f"benchmark videos={len(entries)} output={args.output_dir} "
        f"gpu={torch.cuda.get_device_name(device)}"
    )
    summaries: list[dict[str, Any]] = []
    started = time.perf_counter()
    for index, entry in enumerate(entries, start=1):
        print(f"progress={index}/{len(entries)}")
        summaries.append(
            process_video(
                args,
                entry,
                device,
                training_history,
            )
        )
    summary = make_global_summary(args, summaries, training_history)
    summary["elapsed_seconds"] = time.perf_counter() - started
    json_dump(args.output_dir / "benchmark_summary.json", summary)
    print(
        json.dumps(
            {
                "status": summary["status"],
                "videos": len(summaries),
                "metrics": summary["metrics"],
                "delta": summary["drift_trained_minus_yoda"],
                "summary": str(args.output_dir / "benchmark_summary.json"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
