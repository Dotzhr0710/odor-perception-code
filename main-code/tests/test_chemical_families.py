"""Checks for masked family scoring, SMARTS definitions and prediction alignment."""
from pathlib import Path
import sys
import unittest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from real.tools.chemical_families import aggregate, family_memberships, group_metrics, specification, validate_arrays


class FamilyTests(unittest.TestCase):
    def test_missing_and_single_class_labels_are_excluded(self):
        logits = np.array([[-3., 100.], [3., -100.], [-1., 100.], [1., -100.]])
        labels = np.array([[0., 0.], [1., 0.], [0., 1.], [1., 1.]])
        masks = np.array([[1., 1.], [1., 1.], [1., 0.], [1., 0.]])
        score = group_metrics(logits, labels, masks, np.ones(4, dtype=bool))
        self.assertEqual(score["eligible_task_indices"], [0])
        self.assertEqual(score["roc_auc"], 1.)
        self.assertEqual(score["pr_auc"], 1.)
        empty = group_metrics(logits, labels, masks, np.zeros(4, dtype=bool))
        self.assertIsNone(empty["roc_auc"])

    def test_family_definitions_and_overlaps(self):
        rules = specification()["families"]
        self.assertEqual(len(rules), 16)
        smiles = ["CCO", "c1ccccc1O", "CC(=O)N", "CC(=O)OC", "COC", "c1ccccc1N"]
        m = family_memberships(smiles, np.arange(len(smiles)), rules)
        self.assertTrue(m["aliphatic_alcohol"][0])
        self.assertTrue(m["aromatic"][1] and m["phenol"][1])
        self.assertFalse(m["aliphatic_alcohol"][1])
        self.assertTrue(m["amide"][2])
        self.assertFalse(m["amine"][2])
        self.assertTrue(m["ester"][3])
        self.assertFalse(m["ether"][3])
        self.assertTrue(m["ether"][4])
        self.assertTrue(m["aromatic"][5] and m["amine"][5])

    def test_population_standard_deviation(self):
        result = aggregate([{"x": {"roc_auc": 0.}}, {"x": {"roc_auc": 1.}}], "x", "roc_auc")
        self.assertEqual((result["mean"], result["std"]), (.5, .5))

    def test_prediction_order_mask_and_finiteness_guards(self):
        arrays = {"indices": np.array([2, 4]), "task_names": np.array(["odor"]),
                  "logits": np.zeros((2, 1)), "labels": np.array([[0.], [1.]]), "masks": np.ones((2, 1))}
        expected = (arrays["indices"].copy(), arrays["task_names"].copy(), arrays["labels"].copy(), arrays["masks"].copy())
        validate_arrays(arrays, *expected)
        mutations = {"indices": np.array([4, 2]), "task_names": np.array(["different"]),
                     "masks": np.zeros((2, 1)), "logits": np.full((2, 1), np.nan)}
        for key, value in mutations.items():
            with self.subTest(key=key), self.assertRaises(RuntimeError):
                validate_arrays({**arrays, key: value}, *expected)


if __name__ == "__main__":
    unittest.main()
