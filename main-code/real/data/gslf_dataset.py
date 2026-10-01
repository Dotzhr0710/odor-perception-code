"""Row-aligned GS-LF inputs and preprocessing for the selected configuration."""
import hashlib
import numpy as np
import pandas as pd
import torch
from real.tools import runtime as c

HASHES = {"or_logits.pt": "b49a9f25d8f87a3185cb44b1f7cb25ffe76930c2c30f0661df05c36b31fc1e00",
          "geometry.csv": "604fb770c4c12c1e763d895d5478a8da3f043112980699e1d66b3ffaae0b20f8"}


def array_sha(value):
    if torch.is_tensor(value):
        value = value.detach().cpu().numpy()
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def fit_properties(smiles, train_indices):
    from rdkit import Chem
    from rdkit.Chem import Descriptors, Crippen, rdMolDescriptors
    values = []
    for smi in smiles:
        mol = Chem.MolFromSmiles(smi)
        c.require(mol is not None, "Invalid molecular-property SMILES")
        values.append([Descriptors.MolWt(mol), Crippen.MolLogP(mol), rdMolDescriptors.CalcTPSA(mol),
                       rdMolDescriptors.CalcNumHBD(mol), rdMolDescriptors.CalcNumHBA(mol),
                       rdMolDescriptors.CalcNumRotatableBonds(mol), rdMolDescriptors.CalcNumRings(mol)])
    raw = np.asarray(values, dtype=np.float64)
    c.require(np.isfinite(raw).all(), "Non-finite molecular properties")
    train = raw[train_indices]
    median = np.median(train, axis=0)
    q25, q75 = np.percentile(train, [25, 75], axis=0)
    iqr = q75 - q25
    active = iqr >= 1e-6
    scaled = np.zeros_like(raw)
    scaled[:, active] = (raw[:, active] - median[active]) / (iqr[active] + 1e-6)
    scaled = np.clip(scaled, -4., 4.).astype(np.float32)
    return torch.from_numpy(scaled), {"columns": ["MW", "MolLogP", "TPSA", "HBD", "HBA", "RotB", "Ring"],
            "fit_scope": "current_train_only", "median": median.tolist(), "iqr": iqr.tolist(),
            "eps": 1e-6, "clip": 4., "raw_sha256": array_sha(raw), "scaled_sha256": array_sha(scaled)}


def fit_geometry(smiles, frame, train_indices):
    exclude = {"smiles", "canonical_smiles", "source_smiles", "has_3d", "failure_reason", "conf_success"}
    columns = [k for k in frame.columns if k not in exclude]
    c.require(len(columns) == 41, "Expected 41 aggregated geometric descriptors")
    lookup = {}
    for _, row in frame.iterrows():
        for key in ("smiles", "canonical_smiles", "source_smiles"):
            if key in frame and pd.notna(row[key]):
                lookup.setdefault(str(row[key]).strip(), row)
    rows, valid = [], []
    for smi in smiles:
        c.require(smi in lookup, "Missing geometry alignment: " + smi)
        row = lookup[smi]
        vec = pd.to_numeric(row[columns], errors="coerce").to_numpy(dtype=np.float32)
        rows.append(np.nan_to_num(vec, nan=0., posinf=0., neginf=0.))
        valid.append(bool(float(row["has_3d"]) > .5))
    raw = torch.from_numpy(np.stack(rows))
    valid = torch.tensor(valid)
    selected = torch.as_tensor(train_indices, dtype=torch.long)
    selected = selected[valid[selected]]
    c.require(len(selected) > 0, "No valid training geometry")
    mean, std = raw[selected].mean(0), raw[selected].std(0, unbiased=False)
    std = torch.where(std < 1e-6, torch.ones_like(std), std)
    scaled = torch.where(valid[:, None], (raw - mean) / std, torch.zeros_like(raw))
    values = torch.cat((scaled, valid.float()[:, None]), dim=1)
    return values, {"fit_scope": "current_train_valid_geometry_only", "mean": mean.tolist(), "std": std.tolist(),
            "columns": columns + ["has_3d"], "valid_train": len(selected), "valid_total": int(valid.sum()),
            "unmatched_rows": 0, "scaled_sha256": array_sha(values)}


def load_aux(dataset, train_indices):
    for name, digest in HASHES.items():
        c.require(c.sha(c.INPUT_PATHS[name]) == digest, "Auxiliary input checksum mismatch: " + name)
    receptor = torch.load(c.INPUT_PATHS["or_logits.pt"], map_location="cpu")
    if isinstance(receptor, dict):
        tensors = [value for value in receptor.values() if torch.is_tensor(value)]
        c.require(len(tensors) == 1, "Ambiguous OR tensor")
        receptor = tensors[0]
    c.require(tuple(receptor.shape) == (5862, 1237), "OR alignment/shape mismatch")
    receptor = torch.nan_to_num(receptor.float(), nan=0., posinf=30., neginf=-30.)
    train_indices = np.asarray(train_indices, dtype=np.int64)
    properties, prop_info = fit_properties(dataset.smiles, train_indices)
    geometry, geo_info = fit_geometry(dataset.smiles, pd.read_csv(c.INPUT_PATHS["geometry.csv"]), train_indices)
    return {"receptor": receptor, "properties": properties, "geometry": geometry}, {
            "source_hashes": HASHES, "train_indices_sha256": array_sha(train_indices.astype("<i8")),
            "properties": prop_info, "geometry": geo_info,
            "receptor_rows": "original_GS_LF_order_locked_by_dataset_and_OR_SHA256",
            "receptor_sha256": array_sha(receptor), "historical_morvalue_z_used": False}


def historical_properties(smiles, train_indices):
    path = c.INPUT_PATHS["properties"]
    expected = c.read(c.ROOT / "data/configures/training.json")["historical_morvalue_sha256"]
    c.require(c.sha(path) == expected, "Historical M1 input changed")
    columns = ["z_MW", "z_MolLogP", "z_TPSA", "z_HBD", "z_HBA", "z_RotB", "z_Ring"]
    frame = pd.read_csv(path).drop_duplicates("smiles", keep="first").set_index("smiles")
    c.require(all(smi in frame.index for smi in smiles), "Historical M1 SMILES alignment failed")
    aligned = frame.reindex(smiles)
    value = aligned[columns].apply(pd.to_numeric, errors="coerce").fillna(0).to_numpy(dtype=np.float32)
    flags = np.zeros(len(smiles), dtype=np.int64)
    flags[np.asarray(train_indices, dtype=np.int64)] = 1
    mismatch = int(np.count_nonzero(aligned["is_train"].to_numpy() != flags))
    c.require(np.isfinite(value).all(), "Historical property table has infinity")
    return torch.from_numpy(value), {"source_sha256": expected, "columns": columns,
            "fit_scope": "historical_precomputed_z_no_refit", "historical_is_train_mismatch_count": mismatch,
            "scaled_sha256": array_sha(value), "used_for": "submitted_full_model"}


class InputDataset:
    def __init__(self, dataset, indices, arm, aux):
        self.dataset, self.indices, self.arm, self.aux = dataset, indices, arm, aux

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        row = int(self.indices[index])
        return self.dataset[row], row


def make_loaders(dataset, sets, arm, aux):
    from torch.utils.data import DataLoader
    utils, _ = c.historical()

    def collate(samples):
        base = utils.collate_molgraphs([value[0] for value in samples])
        rows = torch.tensor([value[1] for value in samples], dtype=torch.long)
        extra = {}
        if arm != "graph_only":
            extra["receptor_logits"] = aux["receptor"][rows]
        if arm in ("m1_or", "m1_or_3d", "full"):
            extra["properties"] = aux["properties"][rows]
        if arm in ("m1_or_3d", "full"):
            extra["geometry"] = aux["geometry"][rows]
        return (*base, extra)

    return [DataLoader(InputDataset(dataset, part.indices, arm, aux), batch_size=c.config()["batch_size"],
                       shuffle=(i == 0), collate_fn=collate, num_workers=0) for i, part in enumerate(sets)]
