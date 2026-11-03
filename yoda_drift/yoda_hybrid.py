"""YODA pipeline with an interchangeable SANA or Drifting denoiser."""

from __future__ import annotations

from contextlib import nullcontext
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .drift_model import ConditionalDriftGenerator


def _get_config_value(config: Any, key: str) -> Any:
    """Read config values from either mappings or attribute-based objects."""
    if isinstance(config, dict):
        return config[key]
    return getattr(config, key)


def replace_sana_conv_in(transformer: nn.Module) -> None:
    """Attach the extra convolution present in the upstream YODA module.

    This mirrors ``src/YODA.py`` literally. The released diffusers
    ``SanaTransformer2DModel`` uses ``patch_embed`` in ``forward`` and does not
    consume this attribute, so it is retained for checkpoint/layout parity but
    does not alter the transformer computation.
    """
    transformer.conv_in = nn.Conv2d(
        320,
        320,
        kernel_size=(3, 3),
        stride=(1, 1),
        padding=(1, 1),
    )


def sched_one_step(
    scheduler: Any,
    timesteps: torch.Tensor,
    latent_hat: torch.Tensor,
    model_pred: torch.Tensor,
) -> torch.Tensor:
    """Original YODA one-step x0 conversion."""
    alpha_t = scheduler.alphas_cumprod[timesteps]
    sqrt_alpha_t = torch.sqrt(alpha_t)
    sqrt_one_minus_alpha_t = torch.sqrt(1 - alpha_t)
    return (
        latent_hat[:, :32] - sqrt_one_minus_alpha_t * model_pred
    ) / sqrt_alpha_t


class SanaOneStepDenoiser(nn.Module):
    """The original YODA SANA-Sprint DiT + optional LoRA path.

    This class is kept as the regression baseline. It lazily imports
    ``diffusers`` and ``peft`` only when ``from_pretrained`` is used.
    """

    TARGET_MODULES = (
        "to_q",
        "to_k",
        "to_v",
        "to_out.0",
        "ff.net.0.proj",
        "ff.net.2",
        "proj_in",
        "proj_out",
    )

    def __init__(
        self,
        transformer: nn.Module,
        scheduler: Any,
        prompt_embeds: torch.Tensor,
        prompt_attention_mask: torch.Tensor,
        guidance_scale: float = 1.07,
    ) -> None:
        super().__init__()
        self.transformer = transformer
        self.scheduler = scheduler
        self.guidance_scale = float(guidance_scale)
        self.register_buffer("prompt_embeds", prompt_embeds.detach().cpu())
        self.register_buffer(
            "prompt_attention_mask",
            prompt_attention_mask.detach().cpu(),
        )

    @classmethod
    def from_pretrained(
        cls,
        sd_path: str | Path,
        prompt_embeds_path: str | Path,
        prompt_attention_mask_path: str | Path,
        *,
        lora_rank: int = 72,
        use_lora: bool = True,
        guidance_scale: float = 1.07,
        load_sana_weights: bool = True,
    ) -> "SanaOneStepDenoiser":
        """Build the original SANA branch without importing it at module import time.

        YODA's transformer checkpoint contains only LoRA adapter tensors. The
        complete SANA-Sprint transformer must therefore be loaded first and the
        YODA adapter weights applied afterwards. ``load_sana_weights=False`` is
        available only for structural diagnostics and never produces a valid
        YODA denoiser.
        """
        try:
            from diffusers import SanaTransformer2DModel, SCMScheduler
        except ImportError as exc:
            raise RuntimeError(
                "The original SANA branch requires diffusers. "
                "Install the YODA requirements before selecting denoiser='original'."
            ) from exc

        scheduler = SCMScheduler.from_pretrained(str(sd_path), subfolder="scheduler")
        if load_sana_weights:
            transformer = SanaTransformer2DModel.from_pretrained(
                str(sd_path),
                subfolder="transformer",
            )
        else:
            transformer_config = SanaTransformer2DModel.load_config(
                str(sd_path),
                subfolder="transformer",
            )
            transformer = SanaTransformer2DModel.from_config(transformer_config)
        replace_sana_conv_in(transformer)
        if use_lora:
            try:
                from peft import LoraConfig
            except ImportError as exc:
                raise RuntimeError(
                    "LoRA loading requires peft. Pass use_lora=False or install peft."
                ) from exc
            transformer.requires_grad_(False)
            transformer.add_adapter(
                LoraConfig(
                    r=int(lora_rank),
                    init_lora_weights="gaussian",
                    target_modules=list(cls.TARGET_MODULES),
                )
            )

        prompt_embeds = torch.load(prompt_embeds_path, map_location="cpu")
        prompt_attention_mask = torch.load(
            prompt_attention_mask_path,
            map_location="cpu",
        )
        return cls(
            transformer=transformer,
            scheduler=scheduler,
            prompt_embeds=prompt_embeds,
            prompt_attention_mask=prompt_attention_mask,
            guidance_scale=guidance_scale,
        )

    def forward(
        self,
        lq_latent_hat: torch.Tensor,
        ctxs: Sequence[torch.Tensor] | None = None,
        noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run the original single SANA transformer call.

        ``ctxs`` and ``noise`` are accepted for interface parity with the drift
        branch; the original YODA SANA path does not consume them.
        """
        del ctxs, noise
        latent_hat = lq_latent_hat[:, :32]
        batch_size = latent_hat.shape[0]

        guidance = torch.zeros(
            (batch_size,),
            device=latent_hat.device,
            dtype=torch.float32,
        )
        guidance = guidance.to(latent_hat.dtype)
        guidance = guidance * self.transformer.config.guidance_embeds_scale

        if self.prompt_embeds.shape[0] != batch_size:
            prompt_embeds = self.prompt_embeds.expand(batch_size, -1, -1)
            prompt_attention_mask = self.prompt_attention_mask.expand(batch_size, -1)
        else:
            prompt_embeds = self.prompt_embeds
            prompt_attention_mask = self.prompt_attention_mask

        sigma_data = self.scheduler.config.sigma_data
        latents = latent_hat * sigma_data
        latents_model_input = latents / sigma_data
        self.scheduler.set_timesteps(
            1,
            device=latents.device,
            intermediate_timesteps=None,
        )
        timesteps = self.scheduler.timesteps[:-1]
        if hasattr(self.scheduler, "set_begin_index"):
            self.scheduler.set_begin_index(0)

        timestep = timesteps.expand(latents.shape[0])
        scm_timestep = torch.sin(timestep) / (
            torch.cos(timestep) + torch.sin(timestep)
        )
        scm_timestep_expanded = scm_timestep.view(-1, 1, 1, 1)
        latent_model_input = latents_model_input * torch.sqrt(
            scm_timestep_expanded**2 + (1 - scm_timestep_expanded) ** 2
        )
        noise_pred = self.transformer(
            hidden_states=latent_model_input,
            encoder_hidden_states=prompt_embeds.to(
                device=latent_model_input.device,
                dtype=self.transformer.dtype,
            ),
            encoder_attention_mask=prompt_attention_mask.to(
                device=latent_model_input.device
            ),
            timestep=scm_timestep,
            guidance=guidance,
            return_dict=False,
        )[0]
        noise_pred = (
            (1 - 2 * scm_timestep_expanded) * latent_model_input
            + (
                1
                - 2 * scm_timestep_expanded
                + 2 * scm_timestep_expanded**2
            )
            * noise_pred
        ) / torch.sqrt(scm_timestep_expanded**2 + (1 - scm_timestep_expanded) ** 2)
        noise_pred = noise_pred.float() * sigma_data

        _, denoised = self.scheduler.step(
            noise_pred,
            timestep,
            latents,
            return_dict=False,
        )
        return denoised / sigma_data


class YodaHybridCodec(nn.Module):
    """Drop-in YODA inter-frame pipeline with a selectable denoiser.

    The TA-AE, CLC, bitstream and DPB objects are injected unchanged. Thus
    ``codec.add_ref_frame``, ``codec.clear_dpb`` and the original bitstream
    methods continue to live on ``self.codec``.
    """

    def __init__(
        self,
        vae: Any,
        codec: Any,
        denoiser: str | nn.Module = "drift",
        *,
        latent_channels: int = 32,
        context_channels: Sequence[int] = (1024, 512, 512, 256, 128),
        denoiser_kwargs: dict[str, Any] | None = None,
        original_denoiser: nn.Module | None = None,
        compute_dtype: str = "fp32",
    ) -> None:
        super().__init__()
        self.vae = vae
        self.codec = codec
        self.latent_channels = int(latent_channels)
        self.scaling_factor = float(getattr(vae.config, "scaling_factor", 1.0))
        if compute_dtype not in {"fp32", "bf16"}:
            raise ValueError("compute_dtype must be 'fp32' or 'bf16'")
        self.compute_dtype = compute_dtype
        self._compute_torch_dtype = (
            torch.bfloat16 if compute_dtype == "bf16" else torch.float32
        )

        if isinstance(denoiser, nn.Module):
            self.denoiser_name = "custom"
            self.denoiser = denoiser
        elif denoiser == "drift":
            self.denoiser_name = "drift"
            kwargs = dict(denoiser_kwargs or {})
            kwargs.setdefault("latent_channels", self.latent_channels)
            kwargs.setdefault("context_channels", tuple(context_channels))
            self.denoiser = ConditionalDriftGenerator(**kwargs)
        elif denoiser == "original":
            self.denoiser_name = "original"
            if original_denoiser is None:
                raise ValueError(
                    "denoiser='original' requires an original_denoiser. "
                    "Use SanaOneStepDenoiser.from_pretrained(...) or "
                    "YodaHybridCodec.from_pretrained(...)."
                )
            self.denoiser = original_denoiser
        else:
            raise ValueError("denoiser must be 'drift', 'original', or an nn.Module")

    @classmethod
    def from_pretrained(
        cls,
        sd_path: str | Path,
        checkpoint_path: str | Path,
        prompt_embeds_path: str | Path | None = None,
        prompt_attention_mask_path: str | Path | None = None,
        *,
        denoiser: str = "drift",
        upstream_root: str | Path = "upstream/YODA-main",
        device: str | torch.device = "cpu",
        lora_rank: int = 72,
        context_channels: Sequence[int] = (1024, 512, 512, 256, 128),
        denoiser_kwargs: dict[str, Any] | None = None,
        load_sana_weights: bool = True,
        compute_dtype: str = "fp32",
    ) -> "YodaHybridCodec":
        """Build the real YODA models and load a local YODA checkpoint.

        This method requires the original YODA dependencies and the native
        entropy coder only for bitstream operations. Algorithmic smoke tests can
        use dependency injection instead.
        """
        import sys

        root = Path(upstream_root).resolve()
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        try:
            from src.CLC import DMC
            from src.TA_AE import AutoencoderTA_AE
            from utils.utils import get_state_dict
        except ImportError as exc:
            raise RuntimeError(
                "Could not import the YODA models. Install its requirements and "
                "set upstream_root to the extracted YODA repository."
            ) from exc

        config = AutoencoderTA_AE.load_config(str(sd_path), subfolder="vae")
        vae = AutoencoderTA_AE.from_config(config)
        codec = DMC()

        if denoiser == "original":
            if prompt_embeds_path is None or prompt_attention_mask_path is None:
                raise ValueError(
                    "prompt_embeds_path and prompt_attention_mask_path are required "
                    "for the original SANA denoiser"
                )
            original = SanaOneStepDenoiser.from_pretrained(
                sd_path,
                prompt_embeds_path,
                prompt_attention_mask_path,
                lora_rank=lora_rank,
                use_lora=True,
                load_sana_weights=load_sana_weights,
            )
        elif denoiser == "drift":
            original = None
        else:
            raise ValueError("pretrained denoiser must be 'drift' or 'original'")

        model = cls(
            vae=vae,
            codec=codec,
            denoiser=denoiser,
            latent_channels=_get_config_value(config, "latent_channels"),
            context_channels=context_channels,
            denoiser_kwargs=denoiser_kwargs,
            original_denoiser=original,
            compute_dtype=compute_dtype,
        )
        checkpoint = get_state_dict(checkpoint_path)
        model.load_report = model._load_matching_weights(checkpoint)
        return model.to(device)

    def _load_matching_weights(self, checkpoint: dict[str, Any]) -> dict[str, int]:
        loaded: dict[str, int] = {}
        modules: list[tuple[str, nn.Module, dict[str, torch.Tensor]]] = [
            ("vae", self.vae, checkpoint.get("state_dict_vae", {})),
            ("codec", self.codec, checkpoint.get("state_dict_codec", {})),
        ]
        if self.denoiser_name == "original":
            modules.append(
                (
                    "denoiser",
                    self.denoiser.transformer,
                    checkpoint.get("state_dict_transformer", {}),
                )
            )
        else:
            modules.append(
                (
                    "denoiser",
                    self.denoiser,
                    checkpoint.get("state_dict_denoiser", {}),
                )
            )

        for name, module, weights in modules:
            module_weights = module.state_dict()
            matching = {
                key: value
                for key, value in weights.items()
                if key in module_weights and value.shape == module_weights[key].shape
            }
            module.load_state_dict(matching, strict=False)
            loaded[name] = len(matching)
        return loaded

    def _autocast(self, device: torch.device) -> Any:
        if device.type == "cuda" and self._compute_torch_dtype == torch.bfloat16:
            return torch.autocast(
                device_type="cuda",
                dtype=self._compute_torch_dtype,
            )
        return nullcontext()

    def _encode(self, image: torch.Tensor, last_recon: torch.Tensor) -> tuple[Any, torch.Tensor]:
        with self._autocast(image.device):
            ctxs = self.vae.decoder.feature_extractor(last_recon)
            latent = self.vae.encode(image, ctxs).latent * self.scaling_factor
        # The native entropy codec stays in FP32 to preserve its exact interface.
        return ctxs, latent.float()

    def _decode(self, latent: torch.Tensor, ctxs: Sequence[torch.Tensor], clamp: bool) -> torch.Tensor:
        with self._autocast(latent.device):
            image = self.vae.decode(
                latent.float() / self.scaling_factor,
                ctxs,
                return_dict=False,
            )
        if isinstance(image, (tuple, list)):
            image = image[0]
        if clamp:
            image = image.clamp(-1, 1)
        return image.float()

    def _denoise(
        self,
        lq_latent_hat: torch.Tensor,
        ctxs: Sequence[torch.Tensor],
        noise: torch.Tensor | None,
    ) -> torch.Tensor:
        with self._autocast(lq_latent_hat.device):
            denoised = self.denoiser(
                lq_latent_hat[:, : self.latent_channels].float(),
                ctxs,
                noise=noise,
            )
        return denoised.float()

    def forward(
        self,
        c_t: torch.Tensor,
        last_recon: torch.Tensor,
        noise: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        ctxs, lq_latent = self._encode(c_t, last_recon)
        codec_output = self.codec(lq_latent)
        if not isinstance(codec_output, (tuple, list)) or len(codec_output) != 4:
            raise ValueError("YODA codec forward must return four values")
        lq_latent_hat, feature, bpp_y, bpp_z = codec_output
        x_denoised = self._denoise(lq_latent_hat, ctxs, noise)
        output_image = self._decode(x_denoised, ctxs, clamp=False)
        return {
            "output_image": output_image,
            "bpp_y": bpp_y,
            "bpp_z": bpp_z,
            "mse_latent": F.mse_loss(x_denoised, lq_latent),
            "x_denoised": x_denoised,
            "lq_latent": lq_latent,
            "lq_latent_hat": lq_latent_hat,
            "ctxs": ctxs,
            "feature": feature,
        }

    def compress(
        self,
        c_t: torch.Tensor,
        last_recon: torch.Tensor,
    ) -> dict[str, Any]:
        ctxs, lq_latent = self._encode(c_t, last_recon)
        compressed = self.codec.compress(lq_latent)
        if not isinstance(compressed, (tuple, list)) or len(compressed) != 2:
            raise ValueError("YODA codec compress must return (lq_latent_hat, bit_stream)")
        lq_latent_hat, bit_stream = compressed
        x_denoised = self._denoise(lq_latent_hat, ctxs, None)
        return {
            "bit_stream": bit_stream,
            "x_hat": self._decode(x_denoised, ctxs, clamp=True),
            "x_denoised": x_denoised,
            "lq_latent": lq_latent,
            "lq_latent_hat": lq_latent_hat,
            "ctxs": ctxs,
        }

    def decompress(
        self,
        bit_stream: Any,
        sps: Any,
        last_recon: torch.Tensor,
    ) -> dict[str, Any]:
        ctxs = self.vae.decoder.feature_extractor(last_recon)
        lq_latent_hat = self.codec.decompress(bit_stream, sps)
        x_denoised = self._denoise(lq_latent_hat, ctxs, None)
        return {
            "x_hat": self._decode(x_denoised, ctxs, clamp=False),
            "x_denoised": x_denoised,
        }
