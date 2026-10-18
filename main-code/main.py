"""Train and evaluate Full CaReOR, including its matching GCN pretraining."""
import argparse
import os
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parent
os.environ.setdefault("DGLBACKEND", "pytorch")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

def str2bool(value):
    if isinstance(value, bool):
        return value
    val = str(value).strip().lower()
    if val in {'1', 'true', 'yes', 'y', 't'}:
        return True
    if val in {'0', 'false', 'no', 'n', 'f'}:
        return False
    raise ValueError('Could not parse bool value: {}'.format(value))

def unpack_gslf_batch(batch_data):
    if len(batch_data) == 7:
        idxs, smiles, bg, labels, masks, ids, node_masks = batch_data
        idxs = [int(idx) for idx in idxs]
        return idxs, smiles, bg, labels, masks, ids, node_masks
    if len(batch_data) == 5:
        idxs, smiles, bg, labels, masks = batch_data
        idxs = [int(idx) for idx in idxs]
        return idxs, smiles, bg, labels, masks, None, None
    raise ValueError('Unsupported batch format with {} fields.'.format(len(batch_data)))

def extract_state_dict(ckpt_obj):
    if isinstance(ckpt_obj, dict):
        if 'model_state_dict' in ckpt_obj:
            return ckpt_obj['model_state_dict']
        if 'state_dict' in ckpt_obj:
            return ckpt_obj['state_dict']
    return ckpt_obj

def normalize_state_dict_prefix(state_dict):
    if any(str(key).startswith('module.') for key in state_dict.keys()):
        return {
            (str(key)[7:] if str(key).startswith('module.') else str(key)): value
            for key, value in state_dict.items()
        }
    return state_dict

def infer_molor_runtime_hparams(state_dict, prot_dim):
    mol2prot_dim = False
    key_q1 = 'cross_attn.query_transform_tensor1.weight'
    if key_q1 in state_dict and state_dict[key_q1].ndim == 2:
        out_dim_q1 = int(state_dict[key_q1].shape[0])
        mol2prot_dim = out_dim_q1 == int(prot_dim)

    predictor_in_dim, n_tasks = infer_predictor_io_dims(state_dict)
    gnn_attended_feats = None
    if predictor_in_dim is not None and predictor_in_dim > prot_dim:

        gnn_attended_feats = int(predictor_in_dim - prot_dim)
    elif 'feat_norm.weight' in state_dict:
        feat_norm_dim = int(state_dict['feat_norm.weight'].shape[0])
        if feat_norm_dim > prot_dim:
            gnn_attended_feats = int(feat_norm_dim - prot_dim)
    if n_tasks is None:
        n_tasks = 1

    return mol2prot_dim, gnn_attended_feats, n_tasks
def infer_predictor_io_dims(state_dict):
    pairs = []
    for key in state_dict:
        if str(key).startswith("predict.predict.") and str(key).endswith(".weight"):
            pairs.append((int(str(key).split(".")[2]), key))
    if not pairs:
        return None, None
    pairs.sort()
    first, last = state_dict[pairs[0][1]], state_dict[pairs[-1][1]]
    return (int(first.shape[1]) if first.ndim == 2 else None,
            int(last.shape[0]) if last.ndim == 2 else None)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("random", "chemical_cluster"), default="random", help="Fixed random or chemical-cluster partition")
    parser.add_argument("--seeds", default=None, help="Comma-separated training seeds; default 7,32,42,63")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--smoke", action="store_true", help="Two epochs per phase; not scientific evidence")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true", help="Check data, backbone and full-model interfaces")
    mode.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--strict-runtime", action="store_true", help="Require the pinned formal Linux/CUDA runtime")
    mode.add_argument("--evaluate", type=Path, help="Evaluate a selected full-model checkpoint")
    mode.add_argument("--family-report", type=Path, help="Regenerate chemical-family tables and plot from a completed chemical-cluster run")
    parser.add_argument("--or_logits_path", type=Path, default=Path("data/datasets/full_1237_ORs_logits.pt"))
    parser.add_argument("--three_d_feat_path", type=Path, default=Path("runs/three_d/gslf_3d_features.csv"))
    parser.add_argument("--backbone_path", type=Path, help="Optional matching graph_only checkpoint; use {seed} for multiple seeds. Default: train one backbone per seed.")
    parser.add_argument("-rp", "--result-path", type=Path, default=Path("classification_results/full_careor"))
    return parser.parse_args(argv)


def setup(args):
    from real.tools.restore_assets import restore
    restore()
    from real.tools import runtime as c
    c.configure_split(args.split)
    c.INPUT_PATHS.clear()
    c.INPUT_PATHS.update({
        "or_logits.pt": args.or_logits_path.resolve(),
        "geometry.csv": args.three_d_feat_path.resolve()})
    return c


def check_model(dataset, sets, aux, device):
    import torch
    import utils
    from real.tools import runtime as c
    from real.models.careor import CaReORModel
    from real.tools.training import make_staged_model
    _, graph, _, _ = utils.collate_molgraphs([dataset[0], dataset[1]])
    graph = graph.to(device)
    for arm in ("graph_only", "full"):
        c.seed_all(7)
        net = make_staged_model(7, device) if arm == "full" else CaReORModel(arm, 7).to(device)
        net.eval()
        extras = {k: aux[v][:2].to(device) for k, v in (("receptor_logits", "receptor"), ("properties", "properties"), ("geometry", "geometry"))}
        y = net(graph, graph.ndata["h"], **extras)
        c.require(y.shape == (2, 152) and torch.isfinite(y).all(), "Invalid model output")
        if net.use_m1:
            bias = net.m1(extras["properties"])
            c.require(torch.equal(bias, torch.zeros_like(bias)), "M1 initialization must be identity")
        if arm == "full":
            c.require(all(not p.requires_grad for module in (net.gnn, net.readout) for p in module.parameters()),
                      "Full-model graph weights must stay frozen")
        y.sum().backward()
        c.require(any(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in net.parameters() if p.requires_grad), "No finite gradients")
    print("PASS: dataset alignment, split, GCN pretraining, Full CaReOR, identity M1 and finite gradients")


def run_one(args, seed, device, folder):
    import numpy as np
    import torch
    from real.tools import runtime as c
    from real.data.gslf_dataset import load_aux
    from real.tools import training as t
    c.require(not folder.exists(), "Output already exists; choose a new --result-path")
    folder.mkdir(parents=True)
    dataset, sets, cache = c.load_dataset(build=True)
    aux, info = load_aux(dataset, sets[0].indices)
    c.configure_execution()
    backbone = t.prepare_backbone(args.backbone_path, dataset, sets, cache, seed, device, folder, args.smoke)
    execution = c.configure_execution(staged=True)
    net = t.make_staged_model(seed, device, backbone, allow_smoke=args.smoke)
    splits = t.staged_splits(net, dataset, sets, aux, device)
    config = {"arm": "full", "seed": seed, "smoke": args.smoke, "runtime": c.versions(), "execution": execution,
              "split_name": c.SPLIT_NAME, "dataset_sha256": c.protocol()["dataset_sha256"],
              "inputs": {k: c.sha(v) for k, v in c.INPUT_PATHS.items()}, "preprocessing": info,
              "training": t.experiment()["staged"], "split": cache["splits"],
              "task_names": dataset.task_names}
    config["backbone"] = {"seed": seed, "checkpoint_sha256": c.sha(backbone),
                          "graph_state_sha256": t.state_hash(net, ("gnn.", "readout."))}
    c.write(folder / "run_config.json", config)
    best, history, audit = t.train_staged(net, splits, seed, device, folder, args.smoke)
    splits.update(t.staged_splits(net, dataset, sets, aux, device, ("test",)))
    c.write(folder / "phase_audit.json", audit)
    c.write(folder / "selection.json", {"selected": best, "checkpoint_sha256": c.sha(folder / "model.pth")})
    result = {"arm": "full", "seed": seed, "smoke": args.smoke, "split_name": c.SPLIT_NAME,
              "selected": best, "metrics": {}, "predictions": {},
              "run_config_sha256": c.sha(folder / "run_config.json"),
              "checkpoint_sha256": c.sha(folder / "model.pth")}
    for split in ("val", "test"):
        metric, arrays = t.staged_predict(net, splits[split], device)
        result["metrics"][split] = metric
        subset = sets[("train", "val", "test").index(split)]
        path = folder / (split + "_predictions.npz")
        np.savez_compressed(path, indices=np.asarray(subset.indices), task_names=np.asarray(dataset.task_names), **arrays)
        result["predictions"][split] = {"file": path.name, "sha256": c.sha(path)}
    c.write(folder / "result.json", result)
    print("COMPLETE full", seed, result["metrics"], flush=True)
    return result


def evaluate_checkpoint(args, device):
    import json
    import torch
    from real.tools import runtime as c
    from real.tools import training as t
    from real.data.gslf_dataset import load_aux
    folder = args.evaluate.resolve().parent
    config = c.read(folder / "run_config.json")
    selection = c.read(folder / "selection.json")
    c.require(c.sha(args.evaluate) == selection["checkpoint_sha256"], "Checkpoint checksum changed")
    c.require(config["inputs"] == {k: c.sha(v) for k, v in c.INPUT_PATHS.items()}, "Evaluation inputs differ from training")
    c.require(config.get("split_name", "random") == c.SPLIT_NAME, "Evaluation --split differs from training")
    c.require(config.get("dataset_sha256") == c.protocol()["dataset_sha256"], "Evaluation dataset differs from training")
    c.require(config["arm"] == "full", "Evaluation expects a Full CaReOR checkpoint")
    seed = config["seed"]
    c.configure_execution(staged=True)
    c.seed_all(seed)
    dataset, sets, cache = c.load_dataset(build=True)
    c.require(config["split"] == cache["splits"], "Evaluation split indices differ from training")
    aux, preprocessing = load_aux(dataset, sets[0].indices)
    c.require(preprocessing == config["preprocessing"], "Evaluation preprocessing differs from training")
    net = t.make_staged_model(seed, device)
    net.load_state_dict(torch.load(args.evaluate, map_location=device)["model_state_dict"], strict=True)
    c.require(t.state_hash(net, ("gnn.", "readout.")) == config["backbone"]["graph_state_sha256"],
              "Full-model graph weights differ from the selected backbone")
    splits = t.staged_splits(net, dataset, sets, aux, device, ("val", "test"))
    result = {}
    for split in ("val", "test"):
        metric, _ = t.staged_predict(net, splits[split], device)
        result[split] = metric
    print(json.dumps(result, indent=2))
    return result


def main(argv=None):
    args = parse_args(argv)
    os.chdir(ROOT_DIR)
    c = setup(args)
    import torch
    from real.data.gslf_dataset import load_aux
    torch.set_num_threads(1)
    if args.strict_runtime:
        c.versions(strict=True)
    if args.prepare_only:
        c.load_dataset(build=True)
        return
    if args.family_report:
        c.require(args.split == "chemical_cluster", "Family reports require --split chemical_cluster")
        from real.tools.chemical_families import summarize_families
        return summarize_families(args.family_report)
    device = torch.device(args.device)
    c.require(device.type != "cuda" or torch.cuda.is_available(), "CUDA unavailable; use --device cpu for local checks")
    if args.evaluate:
        return evaluate_checkpoint(args, device)
    if args.check:
        dataset, sets, _ = c.load_dataset(build=True)
        aux, _ = load_aux(dataset, sets[0].indices)
        return check_model(dataset, sets, aux, device)
    seeds = [int(value) for value in (args.seeds or ("7" if args.smoke else "7,32,42,63")).split(",")]
    c.require(len(seeds) == len(set(seeds)) and len(seeds) > 0, "Seeds must be distinct")
    output = args.result_path.resolve()
    output.mkdir(parents=True, exist_ok=True)
    from real.tools.reporting import summarize
    results = []
    for seed in seeds:
        results.append(run_one(args, seed, device, output / ("full__seed_" + str(seed))))
    summarize(results, output)
    if args.split == "chemical_cluster":
        from real.tools.chemical_families import summarize_families
        summarize_families(output)


if __name__ == "__main__":
    main()
