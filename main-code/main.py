"""Train and evaluate the TNNLS CaReOR model and component configurations."""
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
    parser.add_argument("--arm", choices=("graph_only", "raw_or", "m1_or", "m1_or_3d", "full"), default="full")
    parser.add_argument("--all-configurations", action="store_true")
    parser.add_argument("--seeds", default=None, help="Comma-separated training seeds; default 7,32,42,63")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--smoke", action="store_true", help="Two epochs per phase; not scientific evidence")
    parser.add_argument("--check", action="store_true", help="Check data, model interfaces and gradients")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--strict-runtime", action="store_true", help="Require the pinned formal Linux/CUDA runtime")
    parser.add_argument("--evaluate", type=Path, help="Evaluate a selected model.pth")
    parser.add_argument("--or_logits_path", type=Path, default=Path("data/datasets/full_1237_ORs_logits.pt"))
    parser.add_argument("--morvalue_csv", type=Path, default=Path("../morvalue.csv"))
    parser.add_argument("--three_d_feat_path", type=Path, default=Path("runs/three_d/gslf_3d_features.csv"))
    parser.add_argument("--backbone_path", type=Path, default=Path("data/checkpoints/graph_encoder_readout.pth"))
    parser.add_argument("-rp", "--result-path", type=Path, default=Path("classification_results/full_careor"))
    return parser.parse_args(argv)


def setup(args):
    from real.tools.restore_assets import restore
    restore()
    from real.tools import runtime as c
    c.INPUT_PATHS.update({
        "or_logits.pt": args.or_logits_path.resolve(),
        "geometry.csv": args.three_d_feat_path.resolve(),
        "properties": args.morvalue_csv.resolve(),
        "backbone": args.backbone_path.resolve()})
    return c


def check_model(dataset, sets, aux, device):
    import torch
    import utils
    from real.tools import runtime as c
    from real.models.careor import CaReORModel
    from real.tools.training import make_staged_model
    _, graph, _, _ = utils.collate_molgraphs([dataset[0], dataset[1]])
    graph = graph.to(device)
    for arm in ("graph_only", "raw_or", "m1_or", "m1_or_3d", "full"):
        c.seed_all(7)
        net = make_staged_model(7, device) if arm == "full" else CaReORModel(arm, 7).to(device)
        net.eval()
        extras = {k: aux[v][:2].to(device) for k, v in (("receptor_logits", "receptor"), ("properties", "properties"), ("geometry", "geometry"))}
        y = net(graph, graph.ndata["h"], **extras)
        c.require(y.shape == (2, 152) and torch.isfinite(y).all(), "Invalid model output")
        if net.use_m1:
            bias = net.m1(extras["properties"])
            c.require(torch.equal(bias, torch.zeros_like(bias)), "M1 initialization must be identity")
        y.sum().backward()
        c.require(any(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in net.parameters() if p.requires_grad), "No finite gradients")
    print("PASS: dataset alignment, split, all five routes, identity M1, finite outputs and gradients")


def run_one(args, arm, seed, device, folder):
    import numpy as np
    import torch
    from real.tools import runtime as c
    from real.data.gslf_dataset import load_aux, historical_properties, make_loaders
    from real.models.careor import CaReORModel
    from real.tools import training as t
    c.require(not folder.exists(), "Output already exists; choose a new --result-path")
    folder.mkdir(parents=True)
    c.seed_all(seed)
    dataset, sets, cache = c.load_dataset(build=True)
    aux, info = ({}, {}) if arm == "graph_only" else load_aux(dataset, sets[0].indices)
    staged = arm == "full"
    if staged:
        aux["properties"], info["properties"] = historical_properties(dataset.smiles, sets[0].indices)
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.deterministic = True
        net = t.make_staged_model(seed, device)
        splits = t.staged_splits(net, dataset, sets, aux, device)
    else:
        torch.use_deterministic_algorithms(False)
        torch.backends.cudnn.deterministic = False
        loaders = make_loaders(dataset, sets, arm, aux)
        net = CaReORModel(arm, seed).to(device)
    config = {"arm": arm, "seed": seed, "smoke": args.smoke, "runtime": c.versions(),
              "inputs": {k: c.sha(v) for k, v in c.INPUT_PATHS.items()}, "preprocessing": info,
              "training": t.experiment()["staged" if staged else "joint"], "split": cache["splits"]}
    c.write(folder / "run_config.json", config)
    if staged:
        best, history, audit = t.train_staged(net, splits, seed, device, folder, args.smoke)
        splits.update(t.staged_splits(net, dataset, sets, aux, device, ("test",)))
    else:
        best, history, audit = t.train_joint(net, loaders[0], loaders[1], device, folder, args.smoke)
    c.write(folder / "selection.json", {"selected": best, "checkpoint_sha256": c.sha(folder / "model.pth")})
    result = {"arm": arm, "seed": seed, "smoke": args.smoke, "selected": best, "metrics": {}}
    for split in ("val", "test"):
        metric, arrays = t.staged_predict(net, splits[split], device) if staged else t.evaluate_loader(net, loaders[("train", "val", "test").index(split)], device)
        result["metrics"][split] = metric
        subset = sets[("train", "val", "test").index(split)]
        np.savez_compressed(folder / (split + "_predictions.npz"), indices=np.asarray(subset.indices), **arrays)
    c.write(folder / "result.json", result)
    print("COMPLETE", arm, seed, result["metrics"], flush=True)
    return result


def evaluate_checkpoint(args, device):
    import json
    import torch
    from real.tools import runtime as c
    from real.tools import training as t
    from real.data.gslf_dataset import load_aux, historical_properties, make_loaders
    from real.models.careor import CaReORModel
    folder = args.evaluate.resolve().parent
    config = c.read(folder / "run_config.json")
    selection = c.read(folder / "selection.json")
    c.require(c.sha(args.evaluate) == selection["checkpoint_sha256"], "Checkpoint checksum changed")
    c.require(config["inputs"] == {k: c.sha(v) for k, v in c.INPUT_PATHS.items()}, "Evaluation inputs differ from training")
    arm, seed = config["arm"], config["seed"]
    c.seed_all(seed)
    dataset, sets, _ = c.load_dataset(build=True)
    aux, _ = ({}, {}) if arm == "graph_only" else load_aux(dataset, sets[0].indices)
    net = t.make_staged_model(seed, device) if arm == "full" else CaReORModel(arm, seed).to(device)
    net.load_state_dict(torch.load(args.evaluate, map_location=device)["model_state_dict"], strict=True)
    if arm == "full":
        aux["properties"], _ = historical_properties(dataset.smiles, sets[0].indices)
        splits = t.staged_splits(net, dataset, sets, aux, device, ("val", "test"))
    else:
        loaders = make_loaders(dataset, sets, arm, aux)
    result = {}
    for split in ("val", "test"):
        metric, _ = t.staged_predict(net, splits[split], device) if arm == "full" else t.evaluate_loader(net, loaders[("train", "val", "test").index(split)], device)
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
    arms = ("graph_only", "raw_or", "m1_or", "m1_or_3d", "full") if args.all_configurations else (args.arm,)
    output = args.result_path.resolve()
    output.mkdir(parents=True, exist_ok=True)
    from real.tools.reporting import summarize
    results = []
    for arm in arms:
        for seed in seeds:
            results.append(run_one(args, arm, seed, device, output / (arm + "__seed_" + str(seed))))
    summarize(results, output)


if __name__ == "__main__":
    main()
