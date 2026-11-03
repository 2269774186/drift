#!/usr/bin/env python3
"""Train the YODA conditional Drifting generator on Vimeo-90K septuplets.

The original YODA intra and inter branches generate the positive latent
distribution online. Only the Drifting generator receives gradients. Each
septuplet contributes six consecutive P-frame transitions after a real YODA
intra frame establishes the initial decoded reference and DPB state.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Sequence

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
UPSTREAM_ROOT = ROOT / "upstream/YODA-main"
if str(UPSTREAM_ROOT) not in sys.path:
    sys.path.insert(0, str(UPSTREAM_ROOT))

from yoda_drift.drift_model import ConditionalDriftGenerator
from yoda_drift.hybrid_io import load_matching
from yoda_drift.trainer import ConditionalDriftTrainer
from yoda_drift.yoda_hybrid import YodaHybridCodec


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("/data1/zy/vimeo90k/vimeo_septuplet"),
    )
    parser.add_argument("--list-name", default="sep_trainlist.txt")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/data1/zy/yoda_drift_full/train"),
    )
    parser.add_argument("--sd-path", type=Path, default=ROOT / "checkpoints/sana-sprint")
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
    parser.add_argument("--max-clips", type=int, default=0)
    parser.add_argument("--limit-steps", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--clip-batch", type=int, default=4)
    parser.add_argument("--context-chunk", type=int, default=4)
    parser.add_argument("--crop-size", type=int, default=256)
    parser.add_argument("--hidden-channels", type=int, default=64)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--num-buckets", type=int, default=128)
    parser.add_argument("--max-positive", type=int, default=16)
    parser.add_argument("--max-negative", type=int, default=16)
    parser.add_argument("--reconstruction-weight", type=float, default=100.0)
    parser.add_argument("--noise-scale", type=float, default=0.0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-every-steps", type=int, default=500)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--lora-rank", type=int, default=72)
    args = parser.parse_args()
    if args.max_clips < 0 or args.limit_steps < 0:
        raise ValueError("max-clips and limit-steps must be non-negative")
    if args.epochs <= 0 or args.clip_batch <= 0 or args.context_chunk <= 0:
        raise ValueError("epochs, clip-batch and context-chunk must be positive")
    if args.crop_size <= 0 or args.crop_size % 32:
        raise ValueError("crop-size must be positive and divisible by 32")
    return args


def json_dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=True, allow_nan=False),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def set_seed(seed: int) -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":16:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


def load_clip_list(data_root: Path, list_name: str, max_clips: int) -> list[str]:
    list_path = data_root / list_name
    if not list_path.is_file():
        raise FileNotFoundError(f"missing Vimeo clip list: {list_path}")
    clips = [
        line.strip()
        for line in list_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if max_clips:
        clips = clips[:max_clips]
    if not clips:
        raise RuntimeError("the selected Vimeo clip list is empty")
    return clips


def load_septuplet(
    sequence_root: Path,
    clip: str,
    crop_size: int,
    rng: np.random.Generator,
) -> torch.Tensor:
    frames: list[np.ndarray] = []
    for frame_index in range(1, 8):
        path = sequence_root / clip / f"im{frame_index}.png"
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"could not read Vimeo frame: {path}")
        frames.append(image)
    height, width = frames[0].shape[:2]
    if any(frame.shape[:2] != (height, width) for frame in frames):
        raise RuntimeError(f"inconsistent frame sizes in {clip}")
    if height < crop_size or width < crop_size:
        raise RuntimeError(
            f"{clip} is {width}x{height}, smaller than crop {crop_size}"
        )
    left = int(rng.integers(0, width - crop_size + 1))
    top = int(rng.integers(0, height - crop_size + 1))
    cropped = [
        frame[top : top + crop_size, left : left + crop_size]
        for frame in frames
    ]
    stacked = np.stack(cropped, axis=0)
    rgb = cv2.cvtColor(
        stacked.reshape(-1, crop_size, 3),
        cv2.COLOR_BGR2RGB,
    ).reshape(7, crop_size, crop_size, 3)
    tensor = torch.from_numpy(rgb.copy()).permute(0, 3, 1, 2).float()
    tensor = tensor.div_(127.5).sub_(1.0)
    return tensor.reshape(7, 3, crop_size, crop_size)


def load_clip_batch(
    sequence_root: Path,
    clips: Sequence[str],
    crop_size: int,
    rng: np.random.Generator,
) -> torch.Tensor:
    return torch.stack(
        [
            load_septuplet(sequence_root, clip, crop_size, rng)
            for clip in clips
        ],
        dim=0,
    )


def intra_forward(model: Any, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the released YODA intra path and return image plus SANA latent."""
    latent = model.vae.encode(image).latent * model.vae.config.scaling_factor
    latent_hat, _, _ = model.codec(latent)
    latent_hat = latent_hat[:, :32]
    batch_size = latent_hat.shape[0]

    if model.prompt_embeds.shape[0] != batch_size:
        prompt_embeds = model.prompt_embeds.expand(batch_size, -1, -1)
        prompt_attention_mask = model.prompt_attention_mask.expand(batch_size, -1)
    else:
        prompt_embeds = model.prompt_embeds
        prompt_attention_mask = model.prompt_attention_mask

    guidance = torch.zeros(
        (batch_size,),
        device=latent_hat.device,
        dtype=torch.float32,
    ).to(latent_hat.dtype)
    guidance = guidance * model.transformer.config.guidance_embeds_scale
    sigma_data = model.sched.config.sigma_data
    latents = latent_hat * sigma_data
    latents_model_input = latents / sigma_data
    model.sched.set_timesteps(1, device=latents.device, intermediate_timesteps=None)
    timesteps = model.sched.timesteps[:-1]
    if hasattr(model.sched, "set_begin_index"):
        model.sched.set_begin_index(0)

    timestep = timesteps.expand(latents.shape[0])
    scm_timestep = torch.sin(timestep) / (
        torch.cos(timestep) + torch.sin(timestep)
    )
    expanded = scm_timestep.view(-1, 1, 1, 1)
    latent_model_input = latents_model_input * torch.sqrt(
        expanded.square() + (1 - expanded).square()
    )
    noise_pred = model.transformer(
        hidden_states=latent_model_input,
        encoder_hidden_states=prompt_embeds.to(
            device=latent_model_input.device,
            dtype=model.transformer.dtype,
        ),
        encoder_attention_mask=prompt_attention_mask.to(
            device=latent_model_input.device
        ),
        timestep=scm_timestep,
        guidance=guidance,
        return_dict=False,
    )[0]
    noise_pred = (
        (1 - 2 * expanded) * latent_model_input
        + (1 - 2 * expanded + 2 * expanded.square()) * noise_pred
    ) / torch.sqrt(expanded.square() + (1 - expanded).square())
    noise_pred = noise_pred.float() * sigma_data
    _, denoised = model.sched.step(
        noise_pred,
        timestep,
        latents,
        return_dict=False,
    )
    denoised = denoised / sigma_data
    output = model.vae.decode(
        denoised / model.vae.config.scaling_factor,
        return_dict=False,
    )[0]
    return output.clamp(-1, 1), denoised[:, :32]


def load_models(
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[Any, YodaHybridCodec]:
    from src.YODA_INTRA import Yoda_Intra
    from utils.utils import get_state_dict

    intra_args = SimpleNamespace(
        timestep_i=999,
        prompt_embeds_path=str(args.prompt_embeds),
        prompt_attention_mask_path=str(args.prompt_attention_mask),
        lora_rank_transformer=args.lora_rank,
        pretrained_i_weights=str(args.intra_checkpoint),
    )
    intra = Yoda_Intra(sd_path=str(args.sd_path), args=intra_args).to(device)
    intra.codec.update()
    intra.eval()
    intra.requires_grad_(False)

    intra_checkpoint = get_state_dict(args.intra_checkpoint)
    intra_load_report = {
        "codec": load_matching(intra.codec, intra_checkpoint["state_dict_codec"]),
        "vae": load_matching(intra.vae, intra_checkpoint["state_dict_vae"]),
        "transformer": load_matching(
            intra.transformer,
            intra_checkpoint["state_dict_transformer"],
        ),
    }
    if intra_load_report["transformer"] < 400:
        raise RuntimeError(
            f"intra LoRA matching is unexpectedly low: {intra_load_report}"
        )

    inter = YodaHybridCodec.from_pretrained(
        args.sd_path,
        args.inter_checkpoint,
        prompt_embeds_path=args.prompt_embeds,
        prompt_attention_mask_path=args.prompt_attention_mask,
        denoiser="original",
        upstream_root=UPSTREAM_ROOT,
        device=device,
        lora_rank=args.lora_rank,
    )
    inter.codec.update()
    inter.eval()
    inter.requires_grad_(False)
    if inter.load_report.get("denoiser", 0) < 400:
        raise RuntimeError(f"inter LoRA matching is unexpectedly low: {inter.load_report}")
    inter.intra_load_report = intra_load_report
    return intra, inter


def extract_contexts_chunked(
    inter: YodaHybridCodec,
    context_images: torch.Tensor,
    chunk_size: int,
) -> list[torch.Tensor]:
    chunks: list[list[torch.Tensor]] = []
    with torch.no_grad():
        for start in range(0, context_images.shape[0], chunk_size):
            values = inter.vae.decoder.feature_extractor(
                context_images[start : start + chunk_size]
            )
            chunks.append([value.detach() for value in values])
    count = len(chunks[0])
    return [
        torch.cat([chunk[index] for chunk in chunks], dim=0)
        for index in range(count)
    ]


def clip_bucket_id(clip: str, num_buckets: int) -> int:
    digest = hashlib.sha256(clip.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little") % num_buckets


def save_checkpoint(
    path: Path,
    generator: ConditionalDriftGenerator,
    optimizer: torch.optim.Optimizer,
    trainer: ConditionalDriftTrainer,
    *,
    steps: int,
    epoch: int,
    next_index: int,
    order: Sequence[str],
    history: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "format": 1,
            "generator": generator.state_dict(),
            "optimizer": optimizer.state_dict(),
            "positive_bank": trainer.positive_bank.bank,
            "positive_ptr": trainer.positive_bank.ptr,
            "positive_count": trainer.positive_bank.count,
            "negative_bank": trainer.negative_bank.bank,
            "negative_ptr": trainer.negative_bank.ptr,
            "negative_count": trainer.negative_bank.count,
            "steps": steps,
            "epoch": epoch,
            "next_index": next_index,
            "order": list(order),
            "history": history,
            "numpy_rng": np.random.get_state(),
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all(),
        },
        temporary,
    )
    os.replace(temporary, path)


def restore_checkpoint(
    path: Path,
    generator: ConditionalDriftGenerator,
    optimizer: torch.optim.Optimizer,
    trainer: ConditionalDriftTrainer,
) -> dict[str, Any]:
    state = torch.load(path, map_location="cpu", weights_only=False)
    generator.load_state_dict(state["generator"])
    optimizer.load_state_dict(state["optimizer"])
    for name in ("positive", "negative"):
        bank = getattr(trainer, f"{name}_bank")
        bank.bank = state[f"{name}_bank"]
        bank.ptr = state[f"{name}_ptr"]
        bank.count = state[f"{name}_count"]
    np.random.set_state(state["numpy_rng"])
    torch.set_rng_state(state["torch_rng"])
    torch.cuda.set_rng_state_all(state["cuda_rng"])
    return state


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device(args.device)
    if device.type != "cuda":
        raise ValueError("use a CUDA device")
    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    clips = load_clip_list(args.data_root, args.list_name, args.max_clips)
    sequence_root = args.data_root / "sequences"
    missing = [
        clip
        for clip in clips
        if not (sequence_root / clip / "im7.png").is_file()
    ]
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} selected clips are missing, first: {missing[0]}"
        )

    rng = np.random.default_rng(args.seed)
    order = list(clips)
    rng.shuffle(order)

    intra, inter = load_models(args, device)
    generator = ConditionalDriftGenerator(
        latent_channels=32,
        context_channels=(1024, 512, 512, 256, 128),
        hidden_channels=args.hidden_channels,
        depth=args.depth,
        noise_scale=args.noise_scale,
    ).to(device)
    generator.train()
    optimizer = torch.optim.AdamW(
        generator.parameters(),
        lr=args.lr,
        weight_decay=1e-4,
    )
    trainer = ConditionalDriftTrainer(
        generator,
        latent_shape=(32, args.crop_size // 32, args.crop_size // 32),
        num_buckets=args.num_buckets,
        max_positive=args.max_positive,
        max_negative=args.max_negative,
        positive_weight=1.0,
        negative_weight=1.0,
        reconstruction_weight=args.reconstruction_weight,
        R_list=(0.2, 0.05, 0.02),
        bank_device="cpu",
    )

    history: dict[str, Any] = {
        "status": "running",
        "dataset": "Vimeo-90K septuplet",
        "list": str(args.data_root / args.list_name),
        "selected_clips": len(clips),
        "epochs": args.epochs,
        "clip_batch": args.clip_batch,
        "generator_samples_per_step": args.clip_batch * 6,
        "crop_size": args.crop_size,
        "hidden_channels": args.hidden_channels,
        "depth": args.depth,
        "lr": args.lr,
        "reconstruction_weight": args.reconstruction_weight,
        "num_buckets": args.num_buckets,
        "max_positive": args.max_positive,
        "max_negative": args.max_negative,
        "positive_source": "yoda_original_sana_x_denoised",
        "condition_source": "yoda_original_lq_latent_hat",
        "updates": [],
    }
    steps = 0
    epoch_start = 0
    next_index = 0
    checkpoint_path = args.output_dir / "trainer_checkpoint.pt"
    if args.resume and checkpoint_path.is_file():
        state = restore_checkpoint(
            checkpoint_path,
            generator,
            optimizer,
            trainer,
        )
        steps = int(state["steps"])
        epoch_start = int(state["epoch"])
        next_index = int(state["next_index"])
        order = list(state["order"])
        history = state["history"]
        history["status"] = "running"
        print(
            f"resume step={steps} epoch={epoch_start + 1} "
            f"next_index={next_index}"
        )

    generator_path = args.output_dir / "generator.pt"
    history_path = args.output_dir / "train_history.json"
    started = time.perf_counter()
    last_saved_steps = steps
    limit_reached = False

    for epoch in range(epoch_start, args.epochs):
        if epoch != epoch_start:
            next_index = 0
            rng.shuffle(order)
        for start in range(next_index, len(order), args.clip_batch):
            if args.limit_steps and steps >= args.limit_steps:
                limit_reached = True
                break
            batch_clips = order[start : start + args.clip_batch]
            clip_frames = load_clip_batch(
                sequence_root,
                batch_clips,
                args.crop_size,
                rng,
            ).to(device, non_blocking=True)
            batch_size = clip_frames.shape[0]

            with torch.inference_mode():
                first_image, first_latent = intra_forward(
                    intra,
                    clip_frames[:, 0],
                )
                inter.codec.clear_dpb()
                inter.codec.set_curr_poc(0)
                inter.codec.add_ref_frame(
                    None,
                    None,
                    first_latent,
                )
                last_recon = first_image.clamp(-1, 1)
                conditions: list[torch.Tensor] = []
                targets: list[torch.Tensor] = []
                context_images: list[torch.Tensor] = []
                for frame_index in range(1, 7):
                    encoded = inter(
                        clip_frames[:, frame_index],
                        last_recon,
                    )
                    conditions.append(
                        encoded["lq_latent_hat"][:, :32].detach()
                    )
                    targets.append(
                        encoded["x_denoised"][:, :32].detach()
                    )
                    context_images.append(last_recon.detach())
                    last_recon = encoded["output_image"].detach().clamp(-1, 1)
            # Concatenate after inference mode so the trainable generator can
            # save convolution inputs for backward without them being
            # classified as inference tensors.
            condition_batch = torch.cat(conditions, dim=0)
            target_batch = torch.cat(targets, dim=0)
            context_batch = torch.cat(context_images, dim=0)
            contexts = extract_contexts_chunked(
                inter,
                context_batch,
                args.context_chunk,
            )

            bucket_ids = torch.tensor(
                [
                    clip_bucket_id(clip, args.num_buckets)
                    for clip in batch_clips
                    for _ in range(6)
                ],
                dtype=torch.long,
            )
            noise = torch.randn_like(condition_batch)
            metrics = trainer.step(
                condition_batch,
                target_batch,
                contexts,
                optimizer,
                noise=noise,
                bucket_ids=bucket_ids,
                max_grad_norm=args.grad_clip,
            )
            steps += 1
            metrics.update(
                {
                    "step": steps,
                    "epoch": epoch,
                    "clip_start": start,
                    "clips": batch_size,
                    "generator_samples": condition_batch.shape[0],
                    "elapsed_seconds": time.perf_counter() - started,
                }
            )
            history["updates"].append(metrics)

            if steps == 1 or steps % 20 == 0:
                processed = start + batch_size - (0 if epoch == epoch_start else 0)
                remaining = max(len(order) - processed, 0)
                rate = processed / max(time.perf_counter() - started, 1e-9)
                eta = remaining / max(rate, 1e-9)
                print(
                    f"train step={steps:06d} epoch={epoch + 1}/{args.epochs} "
                    f"clip={processed:05d}/{len(order):05d} "
                    f"loss={metrics['loss']:.6g} "
                    f"drift={metrics['drift']:.6g} "
                    f"recon={metrics['reconstruction']:.6g} "
                    f"grad={metrics['grad_norm']:.4g} "
                    f"eta={eta / 3600:.2f}h",
                    flush=True,
                )

            if (
                steps - last_saved_steps >= args.save_every_steps
                or steps == 1
                or limit_reached
            ):
                save_checkpoint(
                    checkpoint_path,
                    generator,
                    optimizer,
                    trainer,
                    steps=steps,
                    epoch=epoch,
                    next_index=start + batch_size,
                    order=order,
                    history=history,
                )
                json_dump(history_path, history)
                last_saved_steps = steps
        if limit_reached:
            break
        next_index = 0

    generator.eval()
    torch.save(generator.state_dict(), generator_path)
    elapsed = time.perf_counter() - started
    history["status"] = "limited" if limit_reached else "complete"
    history["steps"] = steps
    history["elapsed_seconds"] = elapsed
    history["steps_per_second"] = steps / max(elapsed, 1e-9)
    tail = min(25, len(history["updates"]))
    if tail:
        history["final_loss_mean"] = float(
            np.mean([item["loss"] for item in history["updates"][-tail:]])
        )
        history["final_reconstruction_mean"] = float(
            np.mean(
                [item["reconstruction"] for item in history["updates"][-tail:]]
            )
        )
    history["peak_cuda_memory_mb"] = torch.cuda.max_memory_allocated(device) / (
        1024**2
    )
    save_checkpoint(
        checkpoint_path,
        generator,
        optimizer,
        trainer,
        steps=steps,
        epoch=args.epochs - 1,
        next_index=len(order),
        order=order,
        history=history,
    )
    json_dump(history_path, history)
    print(
        json.dumps(
            {
                "status": history["status"],
                "steps": steps,
                "elapsed_seconds": elapsed,
                "generator": str(generator_path),
                "checkpoint": str(checkpoint_path),
            },
            indent=2,
        )
    )

    del intra, inter, generator, optimizer, trainer
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
