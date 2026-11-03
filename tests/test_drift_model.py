import unittest

import torch

from yoda_drift.drift_model import ConditionalDriftGenerator


class CountingGenerator(ConditionalDriftGenerator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.calls = 0

    def forward(self, lq_latent_hat, ctxs=None, noise=None):
        self.calls += 1
        return super().forward(lq_latent_hat, ctxs, noise=noise)


class ConditionalDriftGeneratorTest(unittest.TestCase):
    def test_one_forward_and_context_shapes(self) -> None:
        model = CountingGenerator(
            latent_channels=4,
            context_channels=(8, 4, 2),
            hidden_channels=8,
            depth=1,
        )
        latent = torch.randn(2, 4, 3, 5)
        ctxs = [
            torch.randn(2, 8, 3, 5),
            torch.randn(2, 4, 6, 10),
            torch.randn(2, 2, 12, 20),
        ]
        output = model(latent, ctxs)
        self.assertEqual(model.calls, 1)
        self.assertEqual(output.shape, latent.shape)
        self.assertTrue(torch.isfinite(output).all())

    def test_initial_output_matches_residual_identity(self) -> None:
        model = ConditionalDriftGenerator(
            latent_channels=3,
            context_channels=(),
            hidden_channels=8,
            depth=1,
        )
        latent = torch.randn(1, 3, 2, 2)
        self.assertTrue(torch.allclose(model(latent), latent))

    def test_generator_has_no_lora_parameters(self) -> None:
        model = ConditionalDriftGenerator(
            latent_channels=4,
            context_channels=(8,),
            hidden_channels=8,
            depth=1,
        )
        names = [name for name, _ in model.named_parameters()]
        self.assertTrue(names)
        self.assertFalse(any("lora" in name.lower() for name in names))

    def test_rejects_wrong_context_count(self) -> None:
        model = ConditionalDriftGenerator(
            latent_channels=4,
            context_channels=(8,),
            hidden_channels=8,
            depth=1,
        )
        with self.assertRaises(ValueError):
            model(torch.randn(1, 4, 2, 2), [])

    def test_rejects_context_batch_mismatch(self) -> None:
        model = ConditionalDriftGenerator(
            latent_channels=4,
            context_channels=(8,),
            hidden_channels=8,
            depth=1,
        )
        with self.assertRaisesRegex(ValueError, "batch sizes must match"):
            model(
                torch.randn(2, 4, 2, 2),
                [torch.randn(1, 8, 2, 2)],
            )


if __name__ == "__main__":
    unittest.main()
