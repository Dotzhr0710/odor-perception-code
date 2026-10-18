#!/usr/bin/env python3
"""Build a deterministic, label-blind chemical-cluster split for GS-LF.

The split is frozen from molecular structure only.  Odor labels and model
predictions are never read.  Molecules are clustered with Butina over Morgan
fingerprint Tanimoto distance, then whole clusters are assigned to train,
validation, or test with a deterministic size-balancing rule.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from rdkit.ML.Cluster import Butina


DEFAULT_CSV = Path(
    "data/datasets/NaNs_GS_LF_isomeric_SMILES_dedup_odor_filtered.csv"
)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text_lines(values):
    payload = "".join("{}\n".format(value) for value in values).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def sha256_indices(indices):
    array = np.asarray(indices, dtype="<i8")
    return hashlib.sha256(array.tobytes()).hexdigest()


def parse_ratios(value):
    ratios = [float(item.strip()) for item in value.split(",")]
    if len(ratios) != 3 or any(item <= 0 for item in ratios):
        raise ValueError("--ratios must contain three positive values.")
    if not math.isclose(sum(ratios), 1.0, rel_tol=0.0, abs_tol=1.0e-12):
        raise ValueError("--ratios must sum to exactly 1.")
    return ratios


def cluster_molecules(smiles, radius, n_bits, distance_cutoff, use_chirality):
    canonical_smiles = []
    fingerprints = []
    invalid = []
    for index, value in enumerate(smiles):
        molecule = Chem.MolFromSmiles(str(value))
        if molecule is None:
            invalid.append(index)
            continue
        canonical_smiles.append(
            Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)
        )
        fingerprints.append(
            AllChem.GetMorganFingerprintAsBitVect(
                molecule,
                int(radius),
                nBits=int(n_bits),
                useChirality=bool(use_chirality),
            )
        )
    if invalid:
        raise ValueError(
            "Invalid SMILES at dataset indices: {}".format(invalid[:20])
        )

    condensed_distances = []
    for index in range(1, len(fingerprints)):
        similarities = DataStructs.BulkTanimotoSimilarity(
            fingerprints[index],
            fingerprints[:index],
        )
        condensed_distances.extend(1.0 - float(value) for value in similarities)
    raw_clusters = Butina.ClusterData(
        condensed_distances,
        len(fingerprints),
        float(distance_cutoff),
        isDistData=True,
        reordering=True,
    )
    clusters = [tuple(sorted(int(item) for item in cluster)) for cluster in raw_clusters]
    clusters.sort(key=lambda cluster: (-len(cluster), cluster[0]))
    return canonical_smiles, clusters


def assign_clusters(clusters, dataset_size, ratios):
    """Assign whole clusters with deterministic normalized-fill balancing."""
    split_names = ("train", "val", "test")
    targets = {
        name: float(ratio) * int(dataset_size)
        for name, ratio in zip(split_names, ratios)
    }
    counts = {name: 0 for name in split_names}
    assigned_clusters = {name: [] for name in split_names}

    for cluster_id, cluster in enumerate(clusters):
        # The split with the smallest normalized fill receives the next
        # size-sorted cluster.  Stable split order resolves exact ties.
        destination = min(
            split_names,
            key=lambda name: (
                counts[name] / targets[name],
                split_names.index(name),
            ),
        )
        assigned_clusters[destination].append(cluster_id)
        counts[destination] += len(cluster)

    indices = {}
    for split_name in split_names:
        members = []
        for cluster_id in assigned_clusters[split_name]:
            members.extend(clusters[cluster_id])
        indices[split_name] = sorted(members)

    covered = [item for split_name in split_names for item in indices[split_name]]
    if len(covered) != dataset_size or sorted(covered) != list(range(dataset_size)):
        raise AssertionError("Chemical-cluster split does not cover the dataset exactly.")
    if sum(len(value) for value in indices.values()) != dataset_size:
        raise AssertionError("Chemical-cluster split has overlapping indices.")
    return indices, assigned_clusters, targets


def main():
    parser = argparse.ArgumentParser(
        description="Freeze a deterministic label-blind GS-LF chemical-cluster split."
    )
    parser.add_argument("--dataset-csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/splits/chemical_cluster_rebuilt.json"),
    )
    parser.add_argument("--smiles-column", default="IsomericSMILES")
    parser.add_argument("--radius", type=int, default=2)
    parser.add_argument("--n-bits", type=int, default=2048)
    parser.add_argument(
        "--distance-cutoff",
        type=float,
        default=0.6,
        help="Butina Tanimoto distance cutoff; similarity threshold is 1-cutoff.",
    )
    parser.add_argument("--use-chirality", dest="use_chirality", action="store_true")
    parser.add_argument("--no-chirality", dest="use_chirality", action="store_false")
    parser.set_defaults(use_chirality=True)
    parser.add_argument("--ratios", default="0.8,0.1,0.1")
    parser.add_argument("--expected-rows", type=int, default=5862)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing manifest. Formal use should normally omit this.",
    )
    args = parser.parse_args()

    dataset_csv = args.dataset_csv.resolve()
    output = args.output.resolve()
    if not dataset_csv.is_file():
        raise FileNotFoundError(dataset_csv)
    if output.exists() and not args.overwrite:
        raise FileExistsError(
            "{} already exists; refusing to alter the frozen split.".format(output)
        )
    ratios = parse_ratios(args.ratios)
    table = pd.read_csv(dataset_csv, usecols=[args.smiles_column])
    if len(table) != int(args.expected_rows):
        raise ValueError(
            "Expected {} rows, found {}.".format(args.expected_rows, len(table))
        )
    smiles = table[args.smiles_column].astype(str).tolist()
    canonical_smiles, clusters = cluster_molecules(
        smiles,
        args.radius,
        args.n_bits,
        args.distance_cutoff,
        args.use_chirality,
    )
    indices, assigned_clusters, targets = assign_clusters(
        clusters,
        len(smiles),
        ratios,
    )

    cluster_ids_by_index = [-1] * len(smiles)
    for cluster_id, cluster in enumerate(clusters):
        for index in cluster:
            cluster_ids_by_index[index] = cluster_id
    if min(cluster_ids_by_index) < 0:
        raise AssertionError("At least one molecule was not assigned to a cluster.")

    sizes = np.asarray([len(cluster) for cluster in clusters], dtype=np.int64)
    split_payload = {}
    cluster_split = {}
    for split_name in ("train", "val", "test"):
        split_indices = indices[split_name]
        split_cluster_ids = assigned_clusters[split_name]
        split_payload[split_name] = {
            "size": len(split_indices),
            "fraction": len(split_indices) / len(smiles),
            "indices": split_indices,
            "indices_sha256_int64": sha256_indices(split_indices),
            "cluster_count": len(split_cluster_ids),
            "cluster_ids": split_cluster_ids,
            "cluster_ids_sha256_int64": sha256_indices(split_cluster_ids),
        }
        for cluster_id in split_cluster_ids:
            if str(cluster_id) in cluster_split:
                raise AssertionError(
                    "Cluster {} was assigned to multiple splits.".format(
                        cluster_id
                    )
                )
            cluster_split[str(cluster_id)] = split_name

    expected_cluster_ids = set(range(len(clusters)))
    assigned_cluster_ids = {
        int(cluster_id) for cluster_id in cluster_split
    }
    if assigned_cluster_ids != expected_cluster_ids:
        raise AssertionError(
            "Chemical-cluster split did not assign every cluster exactly once."
        )
    split_cluster_sets = {
        split_name: set(assigned_clusters[split_name])
        for split_name in ("train", "val", "test")
    }
    cluster_leakage_count = sum(
        len(split_cluster_sets[left].intersection(split_cluster_sets[right]))
        for left, right in (
            ("train", "val"),
            ("train", "test"),
            ("val", "test"),
        )
    )
    if cluster_leakage_count != 0:
        raise AssertionError(
            "Chemical-cluster split leaks {} clusters.".format(
                cluster_leakage_count
            )
        )
    all_rows_covered_once = (
        sorted(
            index
            for split_name in ("train", "val", "test")
            for index in indices[split_name]
        )
        == list(range(len(smiles)))
    )
    if not all_rows_covered_once:
        raise AssertionError(
            "Chemical-cluster split does not cover every row exactly once."
        )

    payload = {
        "schema_version": 1,
        "method": "butina_morgan_tanimoto",
        "label_blind": True,
        "predeclared_protocol": {
            "fingerprint": "Morgan bit vector",
            "radius": int(args.radius),
            "n_bits": int(args.n_bits),
            "use_chirality": bool(args.use_chirality),
            "distance": "1 - Tanimoto",
            "distance_cutoff": float(args.distance_cutoff),
            "similarity_threshold": 1.0 - float(args.distance_cutoff),
            "cluster_order": "descending size, then minimum dataset index",
            "assignment": (
                "whole clusters assigned in cluster order to the split with "
                "minimum current_count/target_count; ties train,val,test"
            ),
            "target_ratios": {
                name: ratio
                for name, ratio in zip(("train", "val", "test"), ratios)
            },
            "uses_labels": False,
            "uses_model_outputs": False,
        },
        "dataset": {
            "path": DEFAULT_CSV.as_posix() if dataset_csv.name == DEFAULT_CSV.name else dataset_csv.name,
            "sha256": sha256_file(dataset_csv),
            "size_bytes": dataset_csv.stat().st_size,
            "rows": len(smiles),
            "smiles_column": args.smiles_column,
            "ordered_dataset_indices_sha256_int64": sha256_indices(
                range(len(smiles))
            ),
            "ordered_input_smiles_sha256": sha256_text_lines(smiles),
            "ordered_canonical_smiles_sha256": sha256_text_lines(
                canonical_smiles
            ),
        },
        "clustering": {
            "cluster_count": len(clusters),
            "singleton_count": int(np.sum(sizes == 1)),
            "largest_cluster_size": int(sizes.max()),
            "mean_cluster_size": float(sizes.mean()),
            "median_cluster_size": float(np.median(sizes)),
            "cluster_sizes": sizes.tolist(),
            "cluster_ids_by_dataset_index": cluster_ids_by_index,
            "cluster_ids_by_dataset_index_sha256_int64": sha256_indices(
                cluster_ids_by_index
            ),
            "cluster_to_split": cluster_split,
        },
        "targets": targets,
        "splits": split_payload,
        "invariants": {
            "all_rows_covered_once": all_rows_covered_once,
            "cluster_leakage_count": cluster_leakage_count,
            "labels_read": False,
            "model_outputs_read": False,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print("Chemical-cluster split written to: {}".format(output))
    print(
        "clusters={} singletons={} largest={} split_sizes={}".format(
            len(clusters),
            int(np.sum(sizes == 1)),
            int(sizes.max()),
            {name: len(indices[name]) for name in ("train", "val", "test")},
        )
    )


if __name__ == "__main__":
    main()
