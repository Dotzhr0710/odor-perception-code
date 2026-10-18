"""Regression checks for frozen data partitions (standard library only)."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from real.data.splits import load_cluster_manifest, indices_sha256


class FrozenSplitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.protocol = json.loads((ROOT / "data/configures/protocol.json").read_text())
        cls.dataset = ROOT / cls.protocol["dataset"]
        cls.path = ROOT / cls.protocol["chemical_cluster_split"]["path"]
        cls.manifest = json.loads(cls.path.read_text())

    def validate_mutation(self, manifest):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "split.json"
            path.write_text(json.dumps(manifest))
            return load_cluster_manifest(path, self.dataset)

    def test_original_partition(self):
        m = load_cluster_manifest(self.path, self.dataset, self.protocol["chemical_cluster_split"]["sha256"])
        self.assertEqual([m["splits"][x]["size"] for x in ("train", "val", "test")], [4689, 587, 586])

    def test_modified_dataset_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            dataset = Path(folder) / "data.csv"
            dataset.write_bytes(self.dataset.read_bytes() + b"\n")
            with self.assertRaisesRegex(ValueError, "dataset checksum"):
                load_cluster_manifest(self.path, dataset)

    def test_duplicate_rows_even_with_updated_checksum_are_rejected(self):
        m = copy.deepcopy(self.manifest)
        part = m["splits"]["test"]
        part["indices"][1] = part["indices"][0]
        part["indices_sha256_int64"] = indices_sha256(part["indices"])
        with self.assertRaisesRegex(ValueError, "sorted and unique"):
            self.validate_mutation(m)

    def test_cluster_membership_tampering_is_rejected(self):
        m = copy.deepcopy(self.manifest)
        part = m["splits"]["test"]
        part["cluster_ids"][0] = m["splits"]["train"]["cluster_ids"][0]
        part["cluster_ids_sha256_int64"] = indices_sha256(part["cluster_ids"])
        with self.assertRaisesRegex(ValueError, "membership mismatch"):
            self.validate_mutation(m)

    def test_label_dependent_manifest_is_rejected(self):
        m = copy.deepcopy(self.manifest)
        m["predeclared_protocol"]["uses_labels"] = True
        with self.assertRaisesRegex(ValueError, "protocol changed"):
            self.validate_mutation(m)


if __name__ == "__main__":
    unittest.main()
