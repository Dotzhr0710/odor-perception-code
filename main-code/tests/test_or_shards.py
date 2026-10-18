"""OR inference must preserve every row, receptor column, and source identity."""
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("DGLBACKEND", "pytorch")
import torch
import dgl
import utils
from real.tools import build_gslf_or_logits_sharded as helper
from real.tools.training import save_torch


class SmallDataset(list):
    smiles = ["C", "CC", "CCC"]


class ShardTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(7)

    def test_cpu_inference_covers_multiple_receptors_and_singleton_tail(self):
        from gcn_or_predictor import MolORPredictor
        data = SmallDataset()
        for i, smiles in enumerate(data.smiles):
            graph = dgl.graph(([0, 1], [1, 0]), num_nodes=2)
            graph.ndata["h"] = torch.randn(2, 4)
            data.append((i, smiles, graph, torch.zeros(1), torch.ones(1), None, [1., 1.]))
        batches = [utils.collate_molgraphs(data[:2]), utils.collate_molgraphs(data[2:])]
        net = MolORPredictor(in_feats=4, hidden_feats=[4], prot_feats=4, add_feats=4,
                             max_node_len=2, max_seq_len=2, gnn_attended_feats=4,
                             predictor_hidden_feats=4, n_tasks=1).eval()
        embeddings = torch.randn(2, 2, 4)
        masks = torch.ones(2, 2)
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            cache = folder / "sequences.pt"
            save_torch(cache, {"seq_embeddings": embeddings, "seq_masks": masks})
            save_torch(folder / "model.pth", {"model_state_dict": net.state_dict()})
            args = SimpleNamespace(or_start=0, or_end=2, num_OR_logits=2, output_pt=str(folder / "shard.pt"),
                                   device="cpu", batch_size=2, print_every=1, featurizer_type="canonical",
                                   seq_cache_pt=str(cache), prev_model_path=str(folder))
            with patch.object(helper, 'load_sequence_cache', return_value=(embeddings, masks, 2)), \
                    patch.object(helper, 'load_gslf_dataset', return_value=(data, batches, {"edge_featurizer": None, "featurizer_type": "canonical"}, 2)), \
                    patch.object(helper, 'load_or_model', return_value=net):
                helper.run_shard(args)
            actual = torch.load(args.output_pt)["logits"]
            expected = torch.empty(3, 2)
            with torch.no_grad():
                for i, datum in enumerate(data):
                    graph = datum[2]
                    for j in range(2):
                        expected[i, j] = net(graph, graph.ndata["h"], embeddings[j:j+1], masks[j:j+1], torch.ones(1, 2), torch.device('cpu')).item()
            torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
            self.assertTrue(actual[-1].abs().sum() > 0)
            self.assertTrue(all("h" in batch[2].ndata for batch in batches))

    def payload(self, start, end):
        return {"or_start": start, "or_end": end, "num_OR_logits": 4, "num_molecules": 3,
                "logits": torch.full((3, end-start), float(start + 1)),
                "row_order_sha256": "a"*64, "sequence_cache_sha256": "b"*64, "model_sha256": "c"*64}

    def merge(self, second, folder):
        save_torch(folder / "shard_0.pt", self.payload(0, 2))
        save_torch(folder / "shard_1.pt", second)
        args = SimpleNamespace(shard_glob=str(folder / "shard_*.pt"), num_OR_logits=4, output_pt=str(folder / "merged.pt"))
        helper.run_merge(args)
        return torch.load(args.output_pt)

    def test_merge_preserves_receptor_column_order(self):
        with tempfile.TemporaryDirectory() as temp:
            actual = self.merge(self.payload(2, 4), Path(temp))
            torch.testing.assert_close(actual, torch.tensor([[1., 1., 3., 3.]]).repeat(3, 1))

    def test_merge_rejects_truncated_width_nonfinite_and_mixed_sources(self):
        bad = []
        value = self.payload(2, 4); value['logits'] = torch.ones(3, 1); bad.append(value)
        value = self.payload(2, 4); value['logits'][0, 0] = float('nan'); bad.append(value)
        for key in ('row_order_sha256', 'sequence_cache_sha256', 'model_sha256'):
            value = self.payload(2, 4); value[key] = "d"*64; bad.append(value)
        value = self.payload(2, 4); del value['row_order_sha256']; bad.append(value)
        value = self.payload(1, 4); bad.append(value)
        for index, value in enumerate(bad):
            with self.subTest(case=index), tempfile.TemporaryDirectory() as temp:
                with self.assertRaises(ValueError):
                    self.merge(value, Path(temp))
                self.assertFalse((Path(temp) / "merged.pt").exists())


if __name__ == '__main__':
    unittest.main()
