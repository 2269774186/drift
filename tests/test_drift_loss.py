import unittest

import torch

from yoda_drift.drift_loss import drift_loss, latent_drift_loss


class DriftLossTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(7)
        self.gen = torch.randn(2, 3, 8, requires_grad=True)
        self.pos = torch.randn(2, 4, 8)
        self.neg = torch.randn(2, 2, 8)

    def test_shape_finite_and_gradient(self) -> None:
        loss, info = drift_loss(self.gen, self.pos, self.neg)
        self.assertEqual(loss.shape, (2,))
        self.assertTrue(torch.isfinite(loss).all())
        self.assertIn("loss_0.2", info)
        self.assertIn("scale", info)
        loss.mean().backward()
        self.assertIsNotNone(self.gen.grad)
        self.assertTrue(torch.isfinite(self.gen.grad).all())
        self.assertGreater(float(self.gen.grad.abs().sum()), 0.0)

    def test_loss_without_negatives(self) -> None:
        loss = drift_loss(self.gen, self.pos, return_info=False)
        self.assertEqual(loss.shape, (2,))
        self.assertTrue(torch.isfinite(loss).all())

    def test_targets_are_stop_gradient(self) -> None:
        fixed_pos = self.pos.clone().requires_grad_(True)
        fixed_neg = self.neg.clone().requires_grad_(True)
        loss, _ = drift_loss(self.gen, fixed_pos, fixed_neg)
        loss.mean().backward()
        self.assertIsNone(fixed_pos.grad)
        self.assertIsNone(fixed_neg.grad)

    def test_invalid_temperature(self) -> None:
        with self.assertRaises(ValueError):
            drift_loss(self.gen, self.pos, R_list=(0.2, 0.0))

    def test_latent_loss_collapses_spatial_tokens(self) -> None:
        gen = torch.randn(2, 1, 4, 3, 5, requires_grad=True)
        pos = torch.randn(2, 3, 4, 3, 5)
        neg = torch.randn(2, 2, 4, 3, 5)
        weights = torch.tensor([[1.0, 0.5, 0.25], [1.0, 1.0, 0.0]])
        loss, info = latent_drift_loss(
            gen,
            pos,
            neg,
            weight_pos=weights,
            weight_neg=torch.ones(2, 2),
        )
        self.assertEqual(loss.shape, (2,))
        self.assertTrue(torch.isfinite(loss).all())
        loss.mean().backward()
        self.assertIsNotNone(gen.grad)


if __name__ == "__main__":
    unittest.main()
