"""Validate structure-based split manifests and their row alignment."""
import hashlib
import json
from pathlib import Path
import struct


NAMES = ("train", "val", "test")


def indices_sha256(values):
    return hashlib.sha256(struct.pack("<%dq" % len(values), *values)).hexdigest()


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load_cluster_manifest(path, dataset_path, expected_sha256=None, canonical_smiles=None):
    """Check coverage, hashes, disjoint clusters and the declared Chemical-cluster protocol."""
    path = Path(path)
    if expected_sha256 is not None:
        require(file_sha256(path) == expected_sha256, "Frozen split checksum changed")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    require(manifest["schema_version"] == 1, "Unsupported split schema")
    require(manifest["method"] == "butina_morgan_tanimoto" and manifest["label_blind"] is True,
            "Expected a label-blind chemical-cluster split")
    protocol = manifest["predeclared_protocol"]
    expected = {"radius": 2, "n_bits": 2048, "use_chirality": True,
                "distance_cutoff": 0.6, "uses_labels": False, "uses_model_outputs": False}
    require(all(protocol.get(k) == v for k, v in expected.items()), "Chemical-cluster protocol changed")
    require(protocol["target_ratios"] == {"train": 0.8, "val": 0.1, "test": 0.1},
            "Chemical-cluster target ratios changed")
    dataset = manifest["dataset"]
    require(file_sha256(dataset_path) == dataset["sha256"], "Split dataset checksum mismatch")
    require(Path(dataset_path).stat().st_size == dataset["size_bytes"], "Split dataset size mismatch")
    rows = dataset["rows"]
    require(type(rows) is int and rows > 0, "Invalid split row count")
    require(indices_sha256(list(range(rows))) == dataset["ordered_dataset_indices_sha256_int64"],
            "Dataset index checksum mismatch")
    if canonical_smiles is not None:
        digest = hashlib.sha256("".join(s + "\n" for s in canonical_smiles).encode("utf-8")).hexdigest()
        require(len(canonical_smiles) == rows and digest == dataset["ordered_canonical_smiles_sha256"],
                "Canonical SMILES order differs from the frozen split")
    clustering = manifest["clustering"]
    clusters = clustering["cluster_ids_by_dataset_index"]
    count = clustering["cluster_count"]
    require(len(clusters) == rows and all(type(v) is int and 0 <= v < count for v in clusters),
            "Invalid cluster assignments")
    require(indices_sha256(clusters) == clustering["cluster_ids_by_dataset_index_sha256_int64"],
            "Cluster assignment checksum mismatch")
    sizes = [0] * count
    for cluster in clusters:
        sizes[cluster] += 1
    require(all(sizes) and sizes == clustering["cluster_sizes"], "Cluster sizes mismatch")
    covered, assigned = [], {}
    require(set(manifest["splits"]) == set(NAMES), "Expected train, val and test splits")
    for name in NAMES:
        split = manifest["splits"][name]
        indices, cluster_ids = split["indices"], split["cluster_ids"]
        require(indices and all(type(i) is int and 0 <= i < rows for i in indices), "Invalid split indices")
        require(indices == sorted(set(indices)), "Split indices must be sorted and unique")
        require(len(indices) == split["size"] and indices_sha256(indices) == split["indices_sha256_int64"],
                "Split index count/checksum mismatch: " + name)
        require(len(cluster_ids) == len(set(cluster_ids)) == split["cluster_count"], "Duplicate cluster IDs")
        require(indices_sha256(cluster_ids) == split["cluster_ids_sha256_int64"], "Cluster ID checksum mismatch")
        require(set(cluster_ids) == {clusters[i] for i in indices}, "Split cluster membership mismatch")
        for cluster in cluster_ids:
            require(cluster not in assigned, "A chemical cluster appears in multiple splits")
            assigned[cluster] = name
        covered.extend(indices)
    require(sorted(covered) == list(range(rows)), "Split rows overlap or do not cover the dataset")
    require(set(assigned) == set(range(count)), "Some clusters are missing")
    require({str(k): v for k, v in assigned.items()} == clustering["cluster_to_split"],
            "Cluster-to-split mapping mismatch")
    require(manifest["invariants"] == {"all_rows_covered_once": True, "cluster_leakage_count": 0,
                                       "labels_read": False, "model_outputs_read": False},
            "Split invariants changed")
    return manifest
