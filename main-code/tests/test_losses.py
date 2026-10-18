"""Checks for masked observations in the full-model training objective."""
from pathlib import Path
import sys
import unittest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from real.tools.losses import staged_task_loss
from real.tools.training import experiment


class LossTests(unittest.TestCase):
    def test_unobserved_values_cannot_change_loss_or_gradients(self):
        masks = torch.tensor([[1., 0., 1.], [1., 1., 0.]])
        labels = torch.tensor([[1., 0., 0.], [0., 1., 0.]])
        for split in ("random", "chemical_cluster"):
            before = torch.tensor([[.2, -.1, .5], [.4, -.3, -.8]], requires_grad=True)
            after = before.detach().clone().requires_grad_()
            with torch.no_grad():
                after[masks == 0] = 1e6
            changed_labels = labels.clone()
            changed_labels[masks == 0] = float("nan")
            a = staged_task_loss(before, labels, masks, experiment()["staged_loss"], split)
            b = staged_task_loss(after, changed_labels, masks, experiment()["staged_loss"], split)
            self.assertTrue(torch.equal(a, b))
            a.backward()
            b.backward()
            self.assertTrue(torch.equal(before.grad, after.grad))
            self.assertTrue(torch.equal(after.grad[masks == 0], torch.zeros(2)))

    def test_empty_observations_have_zero_loss_and_gradients(self):
        for split in ("random", "chemical_cluster"):
            logits = torch.ones(3, 2, requires_grad=True)
            value = staged_task_loss(logits, torch.full_like(logits, float("nan")),
                                     torch.zeros_like(logits), experiment()["staged_loss"], split)
            self.assertEqual(value.item(), 0.)
            value.backward()
            self.assertTrue(torch.equal(logits.grad, torch.zeros_like(logits)))


if __name__ == "__main__":
    unittest.main()
