"""Dataset, split, metric and model utilities for CaReOR."""
import hashlib
import json
import os
import platform
from pathlib import Path
import random
import sys
ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("DGLBACKEND", "pytorch")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
INPUT_PATHS = {}

def require(condition, message):
    if not condition:
        raise RuntimeError(message)

def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))

def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp-" + str(os.getpid()))
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(str(tmp), str(path))

def versions(strict=False):
    import torch
    import dgl
    import dgllife
    import numpy
    import pandas
    import scipy
    import sklearn
    import rdkit
    value = {"python": platform.python_version(), "implementation": platform.python_implementation(),
             "torch": torch.__version__, "cuda": torch.version.cuda, "dgl": dgl.__version__,
             "dgllife": dgllife.__version__, "numpy": numpy.__version__, "pandas": pandas.__version__,
             "scipy": scipy.__version__, "sklearn": sklearn.__version__, "rdkit": rdkit.__version__}
    if strict:
        expected = {"python": "3.8.16", "implementation": "CPython", "cuda": "11.8",
                    "dgllife": "0.3.2", "numpy": "1.24.3", "pandas": "2.0.2", "scipy": "1.10.1",
                    "sklearn": "1.2.2", "rdkit": "2023.03.1"}
        for key, wanted in expected.items():
            require(value[key] == wanted, "Runtime mismatch %s: %s != %s" % (key, value[key], wanted))
        require(torch.__version__.split("+")[0] == "2.0.1", "Expected PyTorch 2.0.1")
        require(dgl.__version__ == "1.1.0+cu118", "Expected DGL 1.1.0+cu118")
    return value

def seed_all(seed):
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    # Do not change historical DGL kernels by enabling a new deterministic algorithm policy.

def make_model():
    import utils
    cfg = dict(config(), model="GCN", in_node_feats=74, n_tasks=152)
    model = utils.load_model(cfg)
    audit_model(model)
    return model

def audit_model(model):
    import torch.nn as nn
    layers = model.predict.predict
    require([type(x) for x in layers] == [nn.Dropout, nn.Linear, nn.ReLU, nn.BatchNorm1d, nn.Linear],
            "Classifier structure drift")
    require(tuple(layers[1].weight.shape) == (128, 512), "GCN-only classifier must take 512 features")
    require(tuple(layers[4].weight.shape) == (152, 128), "Classifier output drift")
    require(sum(p.numel() for p in model.predict.parameters()) == 85528, "Head parameter count drift")
    require(all(p.requires_grad for p in model.parameters()), "GCN and classifier must train together")
    require(set(dict(model.named_children())) == {"gnn", "readout", "predict"}, "Unexpected input branch")
    return {"model": "dgllife.model.GCNPredictor", "head_dimensions": [512, 128, 152],
            "head_parameters": 85528, "total_parameters": sum(p.numel() for p in model.parameters()),
            "all_parameters_trainable": True, "inputs": protocol()["inputs"],
            "initialization": protocol()["initialization"]}

def load_dataset(build=False):
    import numpy as np
    import pandas as pd
    import rdkit
    import dgl
    import dgllife
    from rdkit import Chem
    from dgllife.data import MoleculeCSVDataset
    from dgllife.utils import SMILESToBigraph, CanonicalAtomFeaturizer
    p = protocol()
    path = Path(INPUT_PATHS.get("dataset", ROOT / p["dataset"]))
    require(sha(path) == p["dataset_sha256"], "Dataset differs from locked GS-LF source")
    cache = ROOT / "cache/gs_lf_graph_only.bin"
    meta_path = cache.with_suffix(".manifest.json")
    expected = {"dataset_sha256": p["dataset_sha256"], "rdkit": rdkit.__version__,
                "dgl": dgl.__version__, "dgllife": dgllife.__version__,
                "featurizer": "CanonicalAtomFeaturizer", "add_self_loop": True, "node_features": 74}
    exists = cache.exists() or meta_path.exists()
    if exists:
        require(cache.is_file() and meta_path.is_file(), "Incomplete cache; preserve it and inspect before rebuilding")
        meta = read(meta_path)
        require(all(meta.get(k) == v for k, v in expected.items()), "Cache contract mismatch")
        require(sha(cache) == meta["cache_sha256"], "Graph cache checksum mismatch")
    else:
        require(build, "Graph cache missing; run prepare first")
        cache.parent.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(path)
    require(len(df) == p["rows"], "Dataset row count drift")
    # This is the GS_LF preprocessing from historical/m2or.py, without its unrelated OR imports.
    df = df.drop(columns=["Stimulus", "CID", "IUPACName", "MolecularWeight", "name"])
    df["smiles"] = [Chem.MolToSmiles(Chem.MolFromSmiles(s)) for s in df["IsomericSMILES"].tolist()]
    df = df[["smiles"] + [col for col in df.columns if col != "smiles"]].drop(columns=["IsomericSMILES"])
    require(df.shape[1] == p["tasks"] + 1, "Unexpected label columns")
    dataset = MoleculeCSVDataset(df, smiles_to_graph=SMILESToBigraph(
        add_self_loop=True, node_featurizer=CanonicalAtomFeaturizer()),
        smiles_column="smiles", cache_file_path=str(cache), load=exists, n_jobs=1, log_every=1000)
    require(list(dataset.valid_ids) == list(range(p["rows"])), "Invalid molecules or row order change")
    require(dataset.labels.shape == (5862, 152), "Label shape mismatch")
    require(all(g.ndata["h"].shape[1] == 74 for g in dataset.graphs), "Unexpected graph features")
    import utils
    sets = utils.split_dataset({"split": "random", "split_ratio": "0.8,0.1,0.1"}, dataset)
    split_meta = {}
    for name, subset in zip(("train", "val", "test"), sets):
        indices = np.asarray(subset.indices, dtype="<i8")
        digest = hashlib.sha256(indices.tobytes()).hexdigest()
        require(digest == p["split_hashes"][name], "Split indices changed: " + name)
        split_meta[name] = {"size": len(indices), "sha256_int64": digest}
    if not exists:
        write(meta_path, dict(expected, cache_sha256=sha(cache), splits=split_meta,
                              rows=len(dataset), task_names=dataset.task_names))
    else:
        require(meta["splits"] == split_meta and meta["task_names"] == dataset.task_names, "Cache labels/split drift")
    return dataset, sets, read(meta_path)

def metrics(logits, labels, masks):
    import numpy as np
    import torch
    from dgllife.utils import Meter
    meter = Meter()
    meter.update(torch.as_tensor(logits), torch.as_tensor(labels), torch.as_tensor(masks))
    result = {"roc_auc": float(np.mean(meter.compute_metric("roc_auc_score"))),
              "pr_auc": float(np.mean(meter.compute_metric("pr_auc_score")))}
    require(all(np.isfinite(v) for v in result.values()), "Non-finite metric")
    return result


def protocol():
    return read(ROOT / "data/configures/protocol.json")


def config():
    return read(ROOT / "data/configures/GS_LF_OR/GCN_OR_canonical.json")


def historical():
    import utils
    return utils, None
