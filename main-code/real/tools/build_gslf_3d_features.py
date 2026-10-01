import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, rdMolDescriptors

    RDKIT_IMPORT_ERROR = None
except ImportError as exc:
    Chem = None
    AllChem = None
    rdMolDescriptors = None
    RDKIT_IMPORT_ERROR = exc


GEOMETRY_DESCRIPTOR_NAMES = [
    "radius_gyration",
    "asphericity",
    "eccentricity",
    "inertial_shape_factor",
    "npr1",
    "npr2",
    "pmi1",
    "pmi2",
    "pmi3",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Build cached GS-LF 3D descriptor features with RDKit conformers.")
    parser.add_argument("--input_csv", required=True)
    parser.add_argument("--smiles_col", default="smiles")
    parser.add_argument("--output_csv", required=True)
    parser.add_argument("--summary_json", default="")
    parser.add_argument("--num_confs", type=int, default=3)
    parser.add_argument("--max_attempts", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def require_rdkit():
    if RDKIT_IMPORT_ERROR is not None:
        raise ImportError(
            "RDKit is required for build_gslf_3d_features.py. "
            "Please run this script in the project environment from environment.yml."
        ) from RDKIT_IMPORT_ERROR


def canonicalize_smiles(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Failed to parse SMILES: {smiles}")
    return Chem.MolToSmiles(mol)


def load_unique_smiles(input_csv: str, smiles_col: str) -> List[Tuple[str, str]]:
    df = pd.read_csv(input_csv)
    if smiles_col not in df.columns:
        raise ValueError(f"Input CSV missing smiles column: {smiles_col}")

    unique_pairs: Dict[str, str] = {}
    for value in df[smiles_col].fillna("").astype(str).str.strip().tolist():
        if not value:
            continue
        try:
            canonical = canonicalize_smiles(value)
        except Exception:
            canonical = value
        if canonical not in unique_pairs:
            unique_pairs[canonical] = value
    return [(source_smiles, canonical_smiles) for canonical_smiles, source_smiles in unique_pairs.items()]


def make_etkdg_params(seed: int):
    if hasattr(AllChem, "ETKDGv3"):
        params = AllChem.ETKDGv3()
    elif hasattr(AllChem, "ETKDGv2"):
        params = AllChem.ETKDGv2()
    else:
        params = AllChem.ETKDG()
    params.randomSeed = int(seed)
    params.pruneRmsThresh = 0.5
    if hasattr(params, "maxAttempts"):
        params.maxAttempts = 50
    return params


def compute_forcefield_energy(mol_h, conf_id: int) -> float:
    if hasattr(AllChem, "MMFFHasAllMoleculeParams") and AllChem.MMFFHasAllMoleculeParams(mol_h):
        mmff_props = AllChem.MMFFGetMoleculeProperties(mol_h, mmffVariant="MMFF94s")
        if mmff_props is not None:
            AllChem.MMFFOptimizeMolecule(mol_h, mmffVariant="MMFF94s", confId=conf_id, maxIters=200)
            ff = AllChem.MMFFGetMoleculeForceField(mol_h, mmff_props, confId=conf_id)
            if ff is not None:
                return float(ff.CalcEnergy())

    if hasattr(AllChem, "UFFHasAllMoleculeParams") and AllChem.UFFHasAllMoleculeParams(mol_h):
        AllChem.UFFOptimizeMolecule(mol_h, confId=conf_id, maxIters=200)
        uff = AllChem.UFFGetMoleculeForceField(mol_h, confId=conf_id)
        if uff is not None:
            return float(uff.CalcEnergy())

    raise ValueError("Failed to initialize MMFF/UFF force field.")


def compute_geometry_descriptors(mol_h, conf_id: int) -> Dict[str, float]:
    return {
        "radius_gyration": float(rdMolDescriptors.CalcRadiusOfGyration(mol_h, confId=conf_id)),
        "asphericity": float(rdMolDescriptors.CalcAsphericity(mol_h, confId=conf_id)),
        "eccentricity": float(rdMolDescriptors.CalcEccentricity(mol_h, confId=conf_id)),
        "inertial_shape_factor": float(rdMolDescriptors.CalcInertialShapeFactor(mol_h, confId=conf_id)),
        "npr1": float(rdMolDescriptors.CalcNPR1(mol_h, confId=conf_id)),
        "npr2": float(rdMolDescriptors.CalcNPR2(mol_h, confId=conf_id)),
        "pmi1": float(rdMolDescriptors.CalcPMI1(mol_h, confId=conf_id)),
        "pmi2": float(rdMolDescriptors.CalcPMI2(mol_h, confId=conf_id)),
        "pmi3": float(rdMolDescriptors.CalcPMI3(mol_h, confId=conf_id)),
    }


def aggregate_descriptor_rows(descriptor_rows: List[Dict[str, float]], energies: List[float]) -> Dict[str, float]:
    features: Dict[str, float] = {}
    for name in GEOMETRY_DESCRIPTOR_NAMES:
        values = np.asarray([row[name] for row in descriptor_rows], dtype=np.float32)
        features[f"{name}_mean"] = float(values.mean())
        features[f"{name}_min"] = float(values.min())
        features[f"{name}_max"] = float(values.max())
        features[f"{name}_std"] = float(values.std(ddof=0))

    energy_values = np.asarray(energies, dtype=np.float32)
    features["energy_min"] = float(energy_values.min())
    features["energy_mean"] = float(energy_values.mean())
    features["energy_max"] = float(energy_values.max())
    features["energy_std"] = float(energy_values.std(ddof=0))
    features["num_confs_success"] = float(len(energies))
    return features


def zero_feature_dict() -> Dict[str, float]:
    features = {}
    for name in GEOMETRY_DESCRIPTOR_NAMES:
        for stat_name in ["mean", "min", "max", "std"]:
            features[f"{name}_{stat_name}"] = 0.0
    features["energy_min"] = 0.0
    features["energy_mean"] = 0.0
    features["energy_max"] = 0.0
    features["energy_std"] = 0.0
    features["num_confs_success"] = 0.0
    return features


def build_single_row(source_smiles: str, canonical_smiles: str, num_confs: int, seed: int) -> Dict[str, object]:
    base_row = {
        "smiles": canonical_smiles,
        "canonical_smiles": canonical_smiles,
        "source_smiles": source_smiles,
        "has_3d": 0.0,
        "failure_reason": "",
    }

    mol = Chem.MolFromSmiles(canonical_smiles)
    if mol is None:
        return {**base_row, **zero_feature_dict(), "failure_reason": "smiles_parse_failed"}

    mol_h = Chem.AddHs(mol)
    try:
        conf_ids = list(AllChem.EmbedMultipleConfs(mol_h, numConfs=int(num_confs), params=make_etkdg_params(seed)))
    except Exception as exc:
        return {**base_row, **zero_feature_dict(), "failure_reason": f"embed_failed:{type(exc).__name__}"}

    if not conf_ids:
        return {**base_row, **zero_feature_dict(), "failure_reason": "no_conformer_embedded"}

    descriptor_rows = []
    energies = []
    failures = 0
    for conf_id in conf_ids:
        try:
            energy = compute_forcefield_energy(mol_h, conf_id)
            descriptor_rows.append(compute_geometry_descriptors(mol_h, conf_id))
            energies.append(energy)
        except Exception:
            failures += 1

    if not descriptor_rows:
        return {**base_row, **zero_feature_dict(), "failure_reason": "all_forcefield_failed"}

    sorted_pairs = sorted(zip(energies, descriptor_rows), key=lambda item: item[0])[: int(num_confs)]
    ordered_energies = [item[0] for item in sorted_pairs]
    ordered_rows = [item[1] for item in sorted_pairs]

    return {
        **base_row,
        **aggregate_descriptor_rows(ordered_rows, ordered_energies),
        "has_3d": 1.0,
        "failure_reason": "" if failures == 0 else f"partial_failures:{failures}",
    }


def main():
    require_rdkit()
    args = parse_args()

    smiles_pairs = load_unique_smiles(args.input_csv, args.smiles_col)
    rows = []
    failure_count = 0
    partial_failure_count = 0
    failure_reason_counter: Dict[str, int] = {}
    for idx, (source_smiles, canonical_smiles) in enumerate(smiles_pairs, start=1):
        row = build_single_row(source_smiles, canonical_smiles, args.num_confs, args.seed + idx)
        rows.append(row)
        if row["has_3d"] < 0.5:
            failure_count += 1
            reason = str(row["failure_reason"])
            failure_reason_counter[reason] = failure_reason_counter.get(reason, 0) + 1
        if str(row["failure_reason"]).startswith("partial_failures"):
            partial_failure_count += 1
        if idx % 100 == 0 or idx == len(smiles_pairs):
            print(f"[progress] {idx}/{len(smiles_pairs)} molecules processed")

    out_df = pd.DataFrame(rows)
    out_path = Path(args.output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False, encoding="utf-8")

    summary = {
        "num_unique_molecules": len(rows),
        "num_confs_requested": int(args.num_confs),
        "num_has_3d": int((out_df["has_3d"] > 0.5).sum()),
        "num_failed": int(failure_count),
        "num_partial_failures": int(partial_failure_count),
        "failure_ratio": float(failure_count / len(rows)) if rows else 0.0,
        "top_failure_reasons": dict(sorted(failure_reason_counter.items(), key=lambda item: item[1], reverse=True)[:10]),
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    if args.summary_json:
        summary_path = Path(args.summary_json)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        with summary_path.open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
