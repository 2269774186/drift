import unittest

import torch

from yoda_drift.drift_model import ConditionalDriftGenerator
from yoda_drift.trainer import ConditionalDriftTrainer


class ConditionalDriftTrainerTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(11)
        self.generator = ConditionalDriftGenerator(
            latent_channels=2,
            context_channels=(4,),
            hidden_channels=16,
            depth=1,
            noise_scale=0.0,
        )
        self.trainer = ConditionalDriftTrainer(
            self.generator,
            latent_shape=(2, 2, 3),
            max_positive=8,
            max_negative=8,
            reconstruction_weight=0.2,
            R_list=(0.2, 0.05),
        )
        self.condition = torch.randn(2, 2, 2, 3)
        self.context = [torch.randn(2, 4, 2, 3)]
        self.target = self.condition + 0.2

    def test_step_records_history_and_has_finite_gradients(self) -> None:
        optimizer = torch.optim.Adam(self.generator.parameters(), lr=1e-3)
        metrics = self.trainer.step(
            self.condition,
            self.target,
            self.context,
            optimizer,
        )
        self.assertTrue(all(torch.isfinite(torch.tensor(value)) for value in metrics.values()))
        self.assertGreaterEqual(metrics["grad_norm"], 0.0)
        self.assertEqual(self.trainer.positive_bank.available(0), 2)
        self.assertEqual(self.trainer.negative_bank.available(0), 2)
        self.assertTrue(
            all(
                parameter.grad is None or torch.isfinite(parameter.grad).all()
                for parameter in self.generator.parameters()
            )
        )

    def test_loss_decreases_on_fixed_batch(self) -> None:
        optimizer = torch.optim.Adam(self.generator.parameters(), lr=5e-3)
        losses = []
        for _ in range(30):
            metrics = self.trainer.step(
                self.condition,
                self.target,
                self.context,
                optimizer,
            )
            losses.append(metrics["loss"])
        self.assertLess(sum(losses[-5:]) / 5, sum(losses[:5]) / 5)

    def test_rejects_shape_mismatch(self) -> None:
        optimizer = torch.optim.Adam(self.generator.parameters(), lr=1e-3)
        with self.assertRaises(ValueError):
            self.trainer.step(
                self.condition,
                self.target[:1],
                self.context,
                optimizer,
            )

    def test_condition_buckets_keep_per_frame_positives_separate(self) -> None:
        trainer = ConditionalDriftTrainer(
            self.generator,
            latent_shape=(2, 2, 3),
            num_buckets=3,
            max_positive=2,
            max_negative=2,
            reconstruction_weight=1.0,
        )
        optimizer = torch.optim.Adam(self.generator.parameters(), lr=1e-3)
        first_target = self.target + 10
        trainer.step(
            self.condition[:1],
            first_target[:1],
            [self.context[0][:1]],
            optimizer,
            bucket_ids=torch.tensor([0]),
        )
        trainer.step(
            self.condition[1:],
            self.target[1:],
            [self.context[0][1:]],
            optimizer,
            bucket_ids=torch.tensor([2]),
        )
        self.assertEqual(trainer.positive_bank.available(0), 1)
        self.assertEqual(trainer.positive_bank.available(1), 0)
        self.assertEqual(trainer.positive_bank.available(2), 1)

    def test_rejects_bucket_shape_mismatch(self) -> None:
        optimizer = torch.optim.Adam(self.generator.parameters(), lr=1e-3)
        with self.assertRaises(ValueError):
            self.trainer.step(
                self.condition,
                self.target,
                self.context,
                optimizer,
                bucket_ids=torch.tensor([0]),
            )


if __name__ == "__main__":
    unittest.main()
