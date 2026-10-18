"""Regression checks for internal GCN pretraining and split-specific full models."""
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("DGLBACKEND", "pytorch")
import torch
import dgl
import main
from real.models.careor import CaReORModel, ARMS
from real.tools import runtime as c
from real.tools import training as t
from real.data.gslf_dataset import make_loaders, fit_geometry


class ModelTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        c.configure_split("random")
        c.seed_all(7)

    def tearDown(self):
        c.configure_split("random")
        c.configure_execution()

    def graph(self):
        graphs = []
        for _ in range(4):
            graph = dgl.graph(([0, 1, 0, 1], [1, 0, 0, 1]), num_nodes=2)
            graph.ndata["h"] = torch.randn(2, 74)
            graphs.append(graph)
        return dgl.batch(graphs)

    def test_full_model_and_internal_backbone(self):
        self.assertEqual(ARMS, ("graph_only", "full"))
        for arm in ARMS:
            graph = self.graph()
            net = t.make_staged_model(7, torch.device("cpu")) if arm == "full" else CaReORModel(arm, 7)
            net.eval()
            logits = net(graph, graph.ndata["h"], torch.randn(4, 1237), torch.randn(4, 7), torch.randn(4, 42))
            self.assertEqual(tuple(logits.shape), (4, 152))
            self.assertTrue(torch.isfinite(logits).all())
            logits.square().mean().backward()
            if arm == "full":
                self.assertTrue(all(not p.requires_grad and p.grad is None
                                    for module in (net.gnn, net.readout) for p in module.parameters()))
                for module in (net.m1.bias_head, net.m3.or_encoder, net.m3.three_d_encoder, net.predict):
                    self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in module.parameters()))
            else:
                self.assertTrue(all(p.requires_grad for p in net.parameters()))
                self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in net.gnn.parameters()))

    def test_public_entry_runs_full_model_only(self):
        import contextlib
        import io
        args = main.parse_args([])
        self.assertFalse(hasattr(args, "arm"))
        self.assertFalse(hasattr(args, "all_configurations"))
        for flags in (["--arm", "full"], ["--all-configurations"], ["--check", "--prepare-only"]):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                main.parse_args(flags)

    def test_geometry_statistics_use_selected_training_rows(self):
        import pandas as pd
        frame = pd.DataFrame({"smiles": ["C", "CC", "CCC"], "has_3d": [1, 1, 1],
                              **{"f%d" % i: [1., 3., 1000.] for i in range(41)}})
        values, info = fit_geometry(["C", "CC", "CCC"], frame, [0, 1])
        self.assertEqual(info["mean"], [2.] * 41)
        self.assertEqual(info["std"], [1.] * 41)
        self.assertTrue(torch.equal(values[:2, 0], torch.tensor([-1., 1.])))

    def test_chemical_cluster_backbone_provenance_and_smoke_guard(self):
        c.configure_split("chemical_cluster")
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "model.pth"
            path.write_bytes(b"checkpoint validation does not deserialize weights")
            config = {"arm": "graph_only", "seed": 7, "split_name": "chemical_cluster", "dataset_sha256": c.protocol()["dataset_sha256"],
                      "split": {k: {"sha256_int64": v} for k, v in c.expected_split_hashes().items()},
                      "training": t.experiment()["backbone"], "smoke": True}
            c.write(path.parent / "run_config.json", config)
            c.write(path.parent / "selection.json", {"checkpoint_sha256": c.sha(path)})
            t.validate_backbone(path, 7, allow_smoke=True)
            with self.assertRaisesRegex(RuntimeError, "smoke checkpoint"):
                t.validate_backbone(path, 7)
            with self.assertRaisesRegex(RuntimeError, "Backbone seed differs"):
                t.validate_backbone(path, 32, allow_smoke=True)
            config["split_name"] = "random"
            c.write(path.parent / "run_config.json", config)
            with self.assertRaisesRegex(RuntimeError, "same partition"):
                t.validate_backbone(path, 7, allow_smoke=True)

    def test_random_backbone_rejects_cross_seed_and_modified_weights(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "model.pth"
            path.write_bytes(b"original weights")
            config = {"arm": "graph_only", "seed": 7, "split_name": "random",
                      "dataset_sha256": c.protocol()["dataset_sha256"], "smoke": False,
                      "training": t.experiment()["backbone"],
                      "split": {k: {"sha256_int64": v} for k, v in c.expected_split_hashes().items()}}
            c.write(path.parent / "run_config.json", config)
            c.write(path.parent / "selection.json", {"checkpoint_sha256": c.sha(path)})
            t.validate_backbone(path, 7)
            with self.assertRaisesRegex(RuntimeError, "Backbone seed differs"):
                t.validate_backbone(path, 32)
            config["training"]["learning_rate"] = 0.001
            c.write(path.parent / "run_config.json", config)
            with self.assertRaisesRegex(RuntimeError, "training recipe differs"):
                t.validate_backbone(path, 7)
            config["training"] = t.experiment()["backbone"]
            c.write(path.parent / "run_config.json", config)
            path.write_bytes(b"changed weights")
            with self.assertRaisesRegex(RuntimeError, "selected checkpoint"):
                t.validate_backbone(path, 7)

    def test_runtime_policy_matches_both_original_experiments(self):
        for split in ("random", "chemical_cluster"):
            c.configure_split(split)
            for staged in (False, True):
                expected = staged and split == "chemical_cluster"
                c.configure_execution(staged)
                self.assertEqual(torch.are_deterministic_algorithms_enabled(), expected)
                self.assertEqual(torch.backends.cudnn.deterministic, expected)

    def test_staged_updates_leave_frozen_modules_unchanged(self):
        net = t.make_staged_model(7, torch.device("cpu"))
        inputs = (torch.randn(4, 512), torch.randn(4, 1237), torch.randn(4, 7), torch.randn(4, 42))
        active = {"warmup": ("m3.", "predict."), "m1": ("m1.",), "adapt": ("m1.", "m3.", "predict.")}
        for phase, prefixes in active.items():
            t.configure_staged(net, phase)
            before = t.state_copy(net)
            optimizer = torch.optim.AdamW([p for p in net.parameters() if p.requires_grad], lr=1e-3)
            logits = net.forward_features(*inputs)
            logits.square().mean().backward()
            optimizer.step()
            after = net.state_dict()
            changed = {name for name in before if not torch.equal(before[name], after[name])}
            self.assertTrue(changed)
            self.assertTrue(all(name.startswith(prefixes) for name in changed), (phase, changed))
            net.zero_grad(set_to_none=True)


if __name__ == "__main__":
    unittest.main()
