import unittest

import torch

from yoda_drift.memory_bank import TensorMemoryBank


class TensorMemoryBankTest(unittest.TestCase):
    def test_ring_buffer_overwrites_and_samples_without_replacement(self) -> None:
        bank = TensorMemoryBank(2, 3, feature_shape=(2,), device="cpu")
        values = torch.arange(20, dtype=torch.float32).reshape(10, 2)
        buckets = torch.tensor([0, 0, 0, 0, 1, 1], dtype=torch.long)
        bank.add(values[:6], buckets)
        self.assertEqual(bank.available(0), 3)
        self.assertEqual(bank.available(1), 2)

        sampled = bank.sample(torch.tensor([0]), 3, generator=torch.Generator().manual_seed(1))
        self.assertEqual(sampled.shape, (1, 3, 2))
        unique_rows = {tuple(row.tolist()) for row in sampled[0]}
        self.assertEqual(len(unique_rows), 3)

    def test_short_bucket_uses_replacement(self) -> None:
        bank = TensorMemoryBank(1, 4, feature_shape=(1,))
        bank.add(torch.tensor([[5.0]]), torch.tensor([0]))
        sampled = bank.sample(torch.tensor([0, 0]), 3)
        self.assertEqual(sampled.shape, (2, 3, 1))
        self.assertTrue(torch.equal(sampled, torch.full_like(sampled, 5.0)))

    def test_clear_and_empty_bucket(self) -> None:
        bank = TensorMemoryBank(1, 2, feature_shape=(1,))
        with self.assertRaises(RuntimeError):
            bank.sample(torch.tensor([0]), 1)
        bank.add(torch.ones(1, 1), torch.tensor([0]))
        self.assertEqual(bank.available(0), 1)
        bank.clear()
        self.assertEqual(bank.available(0), 0)


if __name__ == "__main__":
    unittest.main()
