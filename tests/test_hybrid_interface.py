import unittest
import inspect
from types import SimpleNamespace

import torch
import torch.nn as nn

from yoda_drift.drift_model import ConditionalDriftGenerator
from yoda_drift.yoda_hybrid import (
    SanaOneStepDenoiser,
    YodaHybridCodec,
    _get_config_value,
    replace_sana_conv_in,
    sched_one_step,
)


class FakeFeatureExtractorModule(nn.Module):
    def __init__(self, context_channels=(8, 4)):
        super().__init__()
        self.context_channels = context_channels

    def forward(self, image):
        batch, _, height, width = image.shape
        return [
            image.new_zeros(batch, channels, max(height // (2**index), 1), max(width // (2**index), 1))
            for index, channels in enumerate(self.context_channels)
        ]


class FakeVAEModule(nn.Module):
    def __init__(self, latent_channels=4, context_channels=(8, 4)):
        super().__init__()
        self.config = SimpleNamespace(scaling_factor=1.0)
        self.decoder = SimpleNamespace(
            feature_extractor=FakeFeatureExtractorModule(context_channels)
        )
        self.latent_channels = latent_channels
        self._encode = nn.Conv2d(3, latent_channels, kernel_size=32, stride=32)
        self._decode = nn.ConvTranspose2d(latent_channels, 3, kernel_size=32, stride=32)

    def encode(self, image, ctxs, return_dict=True):
        del ctxs
        return SimpleNamespace(latent=self._encode(image))

    def decode(self, latent, ctxs, return_dict=False):
        del ctxs, return_dict
        return self._decode(latent)


class FakeCLCModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.forward_calls = 0
        self.compress_calls = 0
        self.decompress_calls = 0
        self.last_sps = None

    def forward(self, latent):
        self.forward_calls += 1
        return (
            latent + 0.1,
            latent,
            torch.ones(latent.shape[0]),
            torch.zeros(latent.shape[0]),
        )

    def compress(self, latent):
        self.compress_calls += 1
        return latent, b"fake-bitstream"

    def decompress(self, bit_stream, sps):
        self.decompress_calls += 1
        self.last_sps = sps
        batch_size = int(sps.get("batch_size", 2))
        return torch.full(
            (batch_size, 4, 2, 2),
            len(bit_stream),
            dtype=torch.float32,
        )


class CountingDenoiser(nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def forward(self, latent, ctxs, noise=None):
        del ctxs, noise
        self.calls += 1
        return latent


class FakeOriginalDenoiser(nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer = nn.Linear(2, 2)

    def forward(self, latent, ctxs=None, noise=None):
        del ctxs, noise
        return latent


class YodaHybridInterfaceTest(unittest.TestCase):
    def setUp(self) -> None:
        self.vae = FakeVAEModule()
        self.clc = FakeCLCModule()
        self.model = YodaHybridCodec(
            self.vae,
            self.clc,
            denoiser="drift",
            latent_channels=4,
            context_channels=(8, 4),
            denoiser_kwargs={"hidden_channels": 8, "depth": 1},
        )
        self.image = torch.rand(2, 3, 64, 64)

    def test_forward_contract(self) -> None:
        output = self.model(self.image, self.image)
        self.assertEqual(output["output_image"].shape, self.image.shape)
        self.assertEqual(output["x_denoised"].shape, (2, 4, 2, 2))
        self.assertEqual(output["bpp_y"].shape, (2,))
        self.assertEqual(self.clc.forward_calls, 1)

    def test_compress_encodes_once_and_denoises_once(self) -> None:
        denoiser = CountingDenoiser()
        model = YodaHybridCodec(
            self.vae,
            self.clc,
            denoiser=denoiser,
            latent_channels=4,
            context_channels=(8, 4),
        )
        output = model.compress(self.image, self.image)
        self.assertEqual(output["bit_stream"], b"fake-bitstream")
        self.assertEqual(output["x_hat"].shape, self.image.shape)
        self.assertEqual(denoiser.calls, 1)
        self.assertEqual(self.clc.compress_calls, 1)

    def test_decompress_contract(self) -> None:
        output = self.model.decompress(
            b"abcd",
            {"ec_part": 0, "batch_size": 1},
            self.image[:1],
        )
        self.assertEqual(output["x_denoised"].shape, (1, 4, 2, 2))
        self.assertEqual(output["x_hat"].shape, (1, 3, 64, 64))
        self.assertEqual(self.clc.decompress_calls, 1)
        self.assertEqual(self.clc.last_sps["batch_size"], 1)

    def test_original_requires_explicit_denoiser(self) -> None:
        with self.assertRaises(ValueError):
            YodaHybridCodec(
                self.vae,
                self.clc,
                denoiser="original",
                latent_channels=4,
                context_channels=(8, 4),
            )

    def test_original_one_step_schedule_matches_yoda_formula(self) -> None:
        scheduler = SimpleNamespace(alphas_cumprod=torch.tensor([0.25]))
        latent_hat = torch.full((1, 32, 2, 2), 2.0)
        model_pred = torch.full((1, 32, 2, 2), 0.5)
        output = sched_one_step(
            scheduler,
            torch.tensor([0]),
            latent_hat,
            model_pred,
        )
        expected = (2.0 - (1 - 0.25) ** 0.5 * 0.5) / 0.25**0.5
        self.assertTrue(torch.allclose(output, torch.full_like(output, expected)))
        self.assertEqual(output.shape, latent_hat.shape)

    def test_original_sana_conv_in_matches_yoda_replacement(self) -> None:
        transformer = SimpleNamespace()
        replace_sana_conv_in(transformer)
        conv_in = transformer.conv_in
        self.assertIsInstance(conv_in, nn.Conv2d)
        self.assertEqual((conv_in.in_channels, conv_in.out_channels), (320, 320))
        self.assertEqual((conv_in.kernel_size, conv_in.stride, conv_in.padding), ((3, 3), (1, 1), (1, 1)))

    def test_original_defaults_require_sana_base_and_yoda_lora_rank(self) -> None:
        sana_defaults = inspect.signature(
            SanaOneStepDenoiser.from_pretrained
        ).parameters
        hybrid_defaults = inspect.signature(
            YodaHybridCodec.from_pretrained
        ).parameters
        self.assertIs(sana_defaults["load_sana_weights"].default, True)
        self.assertIs(hybrid_defaults["load_sana_weights"].default, True)
        self.assertEqual(sana_defaults["lora_rank"].default, 72)
        self.assertEqual(hybrid_defaults["lora_rank"].default, 72)

    def test_drift_branch_is_generator_without_lora(self) -> None:
        self.assertIsInstance(self.model.denoiser, ConditionalDriftGenerator)
        self.assertFalse(
            any("lora" in name.lower() for name, _ in self.model.denoiser.named_parameters())
        )

    def test_original_transformer_checkpoint_keys_load_directly(self) -> None:
        original = FakeOriginalDenoiser()
        model = YodaHybridCodec(
            self.vae,
            self.clc,
            denoiser="original",
            latent_channels=4,
            context_channels=(8, 4),
            original_denoiser=original,
        )
        expected = torch.full_like(original.transformer.weight, 3.0)
        loaded = model._load_matching_weights(
            {"state_dict_transformer": {"weight": expected}}
        )
        self.assertEqual(loaded["denoiser"], 1)
        self.assertTrue(torch.equal(original.transformer.weight, expected))

    def test_config_value_supports_mapping_and_attribute_configs(self) -> None:
        self.assertEqual(_get_config_value({"latent_channels": 32}, "latent_channels"), 32)
        self.assertEqual(
            _get_config_value(SimpleNamespace(latent_channels=32), "latent_channels"),
            32,
        )


if __name__ == "__main__":
    unittest.main()
