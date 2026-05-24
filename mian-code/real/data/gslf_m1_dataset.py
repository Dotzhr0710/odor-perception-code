import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset



REQUIRED_Z_COLUMNS = [
    "z_MW",
    "z_MolLogP",
    "z_TPSA",
    "z_HBD",
    "z_HBA",
    "z_RotB",
    "z_Ring",
]

NON_LABEL_COLUMNS = {
    "smiles",
    "cid",
    "name",
    "iupac_name",
    "iupacname",
    "cas",
    "inchi",
    "inchikey",
    "formula",
    "molecular_weight",
    "split",
    "index",
    "unnamed: 0",
}


THREE_D_META_COLUMNS = {
    "smiles",
    "canonical_smiles",
    "source_smiles",
    "has_3d",
    "conf_success",
    "failure_reason",
}


@dataclass
class GSLFM1Bundle:


    smiles: List[str]
    rdkit_z: torch.Tensor
    logits_base: torch.Tensor
    probs_base: torch.Tensor
    labels: torch.Tensor
    label_mask: torch.Tensor
    label_names: List[str]
    seq_ids: List[str]
    feat_3d_raw: Optional[torch.Tensor] = None
    feat_3d: Optional[torch.Tensor] = None
    has_3d: Optional[torch.Tensor] = None
    three_d_feature_names: Optional[List[str]] = None
    three_d_model_feature_names: Optional[List[str]] = None
    three_d_feature_stats: Optional[Dict[str, float]] = None


class GSLFM1Dataset(Dataset):


    def __init__(self, bundle: GSLFM1Bundle, indices: np.ndarray):
        self.bundle = bundle
        self.indices = np.asarray(indices, dtype=np.int64)

    def __len__(self) -> int:
        return int(self.indices.shape[0])

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        real_idx = int(self.indices[idx])
        return {
            "index": real_idx,
            "smiles": self.bundle.smiles[real_idx],
            "rdkit_z": self.bundle.rdkit_z[real_idx],
            "logits_base": self.bundle.logits_base[real_idx],
            "labels": self.bundle.labels[real_idx],
            "label_mask": self.bundle.label_mask[real_idx],
            **(
                {"feat_3d": self.bundle.feat_3d[real_idx], "has_3d": self.bundle.has_3d[real_idx]}
                if self.bundle.feat_3d is not None and self.bundle.has_3d is not None
                else {}
            ),
        }


def set_global_seed(seed: int) -> None:

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_parent_dir(path: Path) -> None:

    path.parent.mkdir(parents=True, exist_ok=True)


def warn_low_hit_rate(name: str, matched: int, total: int) -> None:

    if total == 0:
        return
    hit_rate = matched / total
    if hit_rate < 0.99:
        print(f"[warning] {name} smiles hit rate is only {hit_rate:.4f} ({matched}/{total}).")


def infer_label_columns(df: pd.DataFrame) -> List[str]:

    label_names: List[str] = []
    for column in df.columns:
        if str(column).strip().lower() in NON_LABEL_COLUMNS:
            continue
        series = df[column]
        numeric = pd.to_numeric(series, errors="coerce")
        if numeric.isna().all():
            continue
        valid = numeric.dropna().unique().tolist()
        if not valid:
            continue
        if all(value in (0, 1, 0.0, 1.0) for value in valid):
            label_names.append(column)
    if not label_names:
        raise ValueError("Could not infer GS-LF label columns from mol_csv.")
    return label_names


def load_mol_dataframe(mol_csv: str) -> Tuple[pd.DataFrame, List[str]]:

    df = pd.read_csv(mol_csv)
    if "smiles" not in df.columns:
        raise ValueError("mol_csv must contain a smiles column.")
    label_names = infer_label_columns(df)
    return df, label_names


def load_morvalue_dataframe(morvalue_csv: str) -> pd.DataFrame:

    df = pd.read_csv(morvalue_csv)
    missing = [column for column in ["smiles"] + REQUIRED_Z_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"morvalue_csv is missing required columns: {missing}")
    return df[["smiles"] + REQUIRED_Z_COLUMNS].copy()


def load_three_d_feature_dataframe(three_d_feat_path: str) -> pd.DataFrame:

    path = Path(three_d_feat_path)
    if not path.exists():
        raise FileNotFoundError(f"3D feature cache not found: {three_d_feat_path}")

    suffix = path.suffix.lower()
    if suffix == ".csv":
        df = pd.read_csv(path)
    elif suffix in {".pkl", ".pickle"}:
        df = pd.read_pickle(path)
    elif suffix in {".pt", ".pth"}:
        payload = torch.load(path, map_location="cpu")
        if isinstance(payload, pd.DataFrame):
            df = payload.copy()
        elif isinstance(payload, dict) and "records" in payload:
            df = pd.DataFrame(payload["records"])
        elif isinstance(payload, list):
            df = pd.DataFrame(payload)
        else:
            raise ValueError(
                "Unsupported 3D .pt payload. Expected DataFrame, list[dict], or dict with 'records'."
            )
    else:
        raise ValueError(f"Unsupported 3D feature file suffix: {path.suffix}")

    if "smiles" not in df.columns and "canonical_smiles" not in df.columns:
        raise ValueError("3D feature cache must contain smiles or canonical_smiles.")
    return df


def infer_three_d_feature_columns(df: pd.DataFrame) -> List[str]:

    feature_names: List[str] = []
    for column in df.columns:
        if str(column).strip().lower() in THREE_D_META_COLUMNS:
            continue
        numeric = pd.to_numeric(df[column], errors="coerce")
        if numeric.isna().all():
            continue
        feature_names.append(column)
    if not feature_names:
        raise ValueError("Could not infer numeric 3D descriptor columns from three_d_feat_path.")
    return feature_names


def build_three_d_alignment(
    smiles_list: List[str],
    three_d_df: pd.DataFrame,
) -> Tuple[torch.Tensor, torch.Tensor, List[str], Dict[str, float]]:

    feature_names = infer_three_d_feature_columns(three_d_df)
    clean_df = three_d_df.copy()

    if "smiles" in clean_df.columns:
        clean_df["smiles"] = clean_df["smiles"].astype(str).str.strip()
    if "canonical_smiles" in clean_df.columns:
        clean_df["canonical_smiles"] = clean_df["canonical_smiles"].astype(str).str.strip()
    if "has_3d" not in clean_df.columns:
        clean_df["has_3d"] = 1.0

    for column in feature_names + ["has_3d"]:
        clean_df[column] = pd.to_numeric(clean_df[column], errors="coerce")
    clean_df["has_3d"] = clean_df["has_3d"].fillna(0.0).clip(lower=0.0, upper=1.0)

    by_smiles = {}
    by_canonical = {}
    for _, row in clean_df.iterrows():

        record = {
            "feat": np.nan_to_num(row[feature_names].to_numpy(dtype=np.float32), nan=0.0),
            "has_3d": float(row["has_3d"]),
        }
        smiles_value = str(row["smiles"]).strip() if "smiles" in clean_df.columns else ""
        canonical_value = str(row["canonical_smiles"]).strip() if "canonical_smiles" in clean_df.columns else ""
        if smiles_value and smiles_value not in by_smiles:
            by_smiles[smiles_value] = record
        if canonical_value and canonical_value not in by_canonical:
            by_canonical[canonical_value] = record

    features = []
    has_3d_values = []
    matched = 0
    valid_3d = 0
    for smiles in smiles_list:

        record = by_smiles.get(smiles)
        if record is None:
            record = by_canonical.get(smiles)
        if record is None:
            features.append(np.zeros(len(feature_names), dtype=np.float32))
            has_3d_values.append([0.0])
            continue
        matched += 1
        if record["has_3d"] > 0.5:
            valid_3d += 1
        features.append(record["feat"])
        has_3d_values.append([record["has_3d"]])

    warn_low_hit_rate("3d_cache", matched, len(smiles_list))
    features_arr = np.stack(features).astype(np.float32, copy=False)
    has_3d_arr = np.asarray(has_3d_values, dtype=np.float32)
    return (
        torch.tensor(features_arr.tolist(), dtype=torch.float32),
        torch.tensor(has_3d_arr.tolist(), dtype=torch.float32),
        feature_names,
        {
            "cache_match_count": float(matched),
            "cache_match_ratio": float(matched / len(smiles_list)) if smiles_list else 0.0,
            "valid_3d_count": float(valid_3d),
            "valid_3d_ratio": float(valid_3d / len(smiles_list)) if smiles_list else 0.0,
        },
    )


def fit_three_d_standardizer(
    feat_3d_raw: torch.Tensor,
    has_3d: torch.Tensor,
    train_indices: np.ndarray,
    feature_names: List[str],
) -> Dict[str, object]:



    train_idx = [int(idx) for idx in np.asarray(train_indices).reshape(-1).tolist()]
    train_feat = feat_3d_raw[train_idx]
    train_mask = has_3d[train_idx].squeeze(1) > 0.5

    if train_mask.any():
        valid_train_feat = train_feat[train_mask]
        mean = valid_train_feat.mean(dim=0)
        std = valid_train_feat.std(dim=0, unbiased=False)
        valid_count = int(valid_train_feat.shape[0])
    else:
        mean = torch.zeros(feat_3d_raw.shape[1], dtype=feat_3d_raw.dtype)
        std = torch.ones(feat_3d_raw.shape[1], dtype=feat_3d_raw.dtype)
        valid_count = 0
        print("[warning] no valid 3D samples found in training split; using identity 3D scaler.")

    std = torch.where(std < 1e-6, torch.ones_like(std), std)
    return {
        "mean": mean.tolist(),
        "std": std.tolist(),
        "feature_names": list(feature_names),
        "valid_train_count": valid_count,
    }


def apply_three_d_standardizer(
    bundle: GSLFM1Bundle,
    scaler_state: Dict[str, object],
    add_success_flag: bool = False,
    fallback_zero: bool = True,
) -> GSLFM1Bundle:

    if bundle.feat_3d_raw is None or bundle.has_3d is None or bundle.three_d_feature_names is None:
        raise ValueError("Bundle does not contain raw 3D features to standardize.")

    mean = torch.tensor(scaler_state["mean"], dtype=bundle.feat_3d_raw.dtype)
    std = torch.tensor(scaler_state["std"], dtype=bundle.feat_3d_raw.dtype)
    if mean.shape[0] != bundle.feat_3d_raw.shape[1]:
        raise ValueError("3D scaler mean dimension does not match feat_3d_raw.")
    if list(scaler_state["feature_names"]) != list(bundle.three_d_feature_names):
        raise ValueError("3D scaler feature_names do not match current 3D cache.")

    feat_3d = (bundle.feat_3d_raw - mean) / std
    if fallback_zero:
        feat_3d = torch.where(bundle.has_3d > 0.5, feat_3d, torch.zeros_like(feat_3d))

    model_feature_names = list(bundle.three_d_feature_names)
    if add_success_flag:
        feat_3d = torch.cat([feat_3d, bundle.has_3d], dim=1)
        model_feature_names.append("has_3d")

    bundle.feat_3d = feat_3d.float()
    bundle.three_d_model_feature_names = model_feature_names
    return bundle


def _tensor_rows_to_smiles(smiles: List[str], tensor: torch.Tensor) -> Dict[str, torch.Tensor]:

    result: Dict[str, torch.Tensor] = {}
    for idx, smiles_value in enumerate(smiles):
        if smiles_value not in result:
            result[smiles_value] = tensor[idx]
    return result


def load_spectrum_payload(spectrum_pt: str) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], List[str]]:

    payload = torch.load(spectrum_pt, map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError("spectrum_pt must be a dict-like torch checkpoint.")
    if "smiles" not in payload or "logits_base" not in payload:
        raise ValueError("spectrum_pt must contain smiles and logits_base.")
    smiles = list(payload["smiles"])
    logits_base = payload["logits_base"].float()
    probs_base = payload.get("probs_base")
    if probs_base is None:
        probs_base = torch.sigmoid(logits_base)
    else:
        probs_base = probs_base.float()
    seq_ids = list(payload.get("seq_ids", []))
    return _tensor_rows_to_smiles(smiles, logits_base), _tensor_rows_to_smiles(smiles, probs_base), seq_ids


def build_gslf_m1_bundle(
    mol_csv: str,
    morvalue_csv: str,
    spectrum_pt: str,
    three_d_feat_path: Optional[str] = None,
) -> GSLFM1Bundle:

    mol_df, label_names = load_mol_dataframe(mol_csv)
    mor_df = load_morvalue_dataframe(morvalue_csv)
    logits_map, probs_map, seq_ids = load_spectrum_payload(spectrum_pt)

    mor_df = mor_df.drop_duplicates(subset=["smiles"], keep="first")
    mol_df = mol_df.drop_duplicates(subset=["smiles"], keep="first")
    merged = mol_df.merge(mor_df, on="smiles", how="left")
    matched_mor = merged[REQUIRED_Z_COLUMNS].notna().all(axis=1).sum()
    warn_low_hit_rate("morvalue", int(matched_mor), len(merged))

    valid_rows = []
    missed_spectrum = 0
    for _, row in merged.iterrows():
        smiles = row["smiles"]
        if smiles not in logits_map:
            missed_spectrum += 1
            continue
        if row[REQUIRED_Z_COLUMNS].isna().any():
            continue
        valid_rows.append(row)
    warn_low_hit_rate("spectrum", len(valid_rows), len(merged))
    if not valid_rows:
        raise ValueError("No aligned GS-LF samples remained after smiles matching.")

    aligned_df = pd.DataFrame(valid_rows).reset_index(drop=True)
    smiles = aligned_df["smiles"].tolist()
    rdkit_z = torch.tensor(aligned_df[REQUIRED_Z_COLUMNS].to_numpy(dtype=np.float32))
    label_mask_np = (~aligned_df[label_names].isna()).to_numpy(dtype=np.float32)
    missing_label_cells = int((label_mask_np == 0).sum())
    if missing_label_cells > 0:
        print(
            f"[data] detected {missing_label_cells} missing label cells; "
            "using author-style label=0 with mask=0."
        )


    labels_np = aligned_df[label_names].fillna(0.0).to_numpy(dtype=np.float32)
    labels = torch.tensor(labels_np)
    label_mask = torch.tensor(label_mask_np)
    logits_base = torch.stack([logits_map[smi] for smi in smiles]).float()
    probs_base = torch.stack([probs_map[smi] for smi in smiles]).float()
    feat_3d_raw = None
    has_3d = None
    three_d_feature_names = None
    three_d_feature_stats = None

    if three_d_feat_path:
        three_d_df = load_three_d_feature_dataframe(three_d_feat_path)
        feat_3d_raw, has_3d, three_d_feature_names, three_d_feature_stats = build_three_d_alignment(
            smiles, three_d_df
        )
        print(
            f"[data] aligned 3D cache with dim={feat_3d_raw.shape[1]}, "
            f"valid_ratio={three_d_feature_stats['valid_3d_ratio']:.4f}"
        )

    print(
        f"[data] aligned {len(smiles)} samples, {labels.shape[1]} labels, "
        f"{logits_base.shape[1]} ORs, missed_spectrum={missed_spectrum}"
    )

    return GSLFM1Bundle(
        smiles=smiles,
        rdkit_z=rdkit_z,
        logits_base=logits_base,
        probs_base=probs_base,
        labels=labels,
        label_mask=label_mask,
        label_names=label_names,
        seq_ids=seq_ids,
        feat_3d_raw=feat_3d_raw,
        has_3d=has_3d,
        three_d_feature_names=three_d_feature_names,
        three_d_feature_stats=three_d_feature_stats,
    )


def random_split_indices(num_samples: int, val_ratio: float, test_ratio: float, seed: int) -> Dict[str, np.ndarray]:

    if num_samples <= 0:
        raise ValueError("num_samples must be positive.")
    if val_ratio < 0 or test_ratio < 0 or (val_ratio + test_ratio) >= 1:
        raise ValueError("Expected val_ratio >= 0, test_ratio >= 0, and val_ratio + test_ratio < 1.")

    rng = np.random.default_rng(seed)
    perm = rng.permutation(num_samples)
    n_test = int(round(num_samples * test_ratio))
    n_val = int(round(num_samples * val_ratio))
    n_test = min(n_test, num_samples - 1)
    n_val = min(n_val, num_samples - n_test - 1)
    test_idx = np.sort(perm[:n_test])
    val_idx = np.sort(perm[n_test:n_test + n_val])
    train_idx = np.sort(perm[n_test + n_val:])
    return {"train": train_idx, "val": val_idx, "test": test_idx}


def split_bundle(bundle: GSLFM1Bundle, val_ratio: float, test_ratio: float, seed: int):

    splits = random_split_indices(len(bundle.smiles), val_ratio=val_ratio, test_ratio=test_ratio, seed=seed)
    return {
        split_name: GSLFM1Dataset(bundle, indices)
        for split_name, indices in splits.items()
    }, splits


def bundle_to_metadata(bundle: GSLFM1Bundle) -> Dict[str, object]:

    return {
        "num_samples": len(bundle.smiles),
        "num_labels": int(bundle.labels.shape[1]),
        "num_ors": int(bundle.logits_base.shape[1]),
        "label_names": bundle.label_names,
        "seq_ids": bundle.seq_ids,
        "three_d_raw_dim": int(bundle.feat_3d_raw.shape[1]) if bundle.feat_3d_raw is not None else 0,
        "three_d_model_dim": int(bundle.feat_3d.shape[1]) if bundle.feat_3d is not None else 0,
        "three_d_feature_names": bundle.three_d_feature_names,
        "three_d_model_feature_names": bundle.three_d_model_feature_names,
        "three_d_feature_stats": bundle.three_d_feature_stats,
    }


def save_json(payload: Dict[str, object], output_path: str) -> None:

    path = Path(output_path)
    ensure_parent_dir(path)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
