"""Chemical-family evaluation using fixed SMARTS and the original metric rules."""
import csv
from pathlib import Path

import numpy as np
from sklearn.metrics import auc, precision_recall_curve, roc_auc_score
from real.tools import runtime as c


def specification():
    return c.read(c.ROOT / "data/configures/chemical_families.json")


def family_memberships(smiles, indices, rules):
    from rdkit import Chem
    molecules = [Chem.MolFromSmiles(smiles[int(i)]) for i in indices]
    c.require(all(mol is not None for mol in molecules), "Invalid test molecule")
    membership = {}
    for rule in rules:
        query = Chem.MolFromSmarts(rule["smarts"])
        c.require(query is not None, "Invalid family SMARTS: " + rule["family"])
        membership[rule["family"]] = np.asarray([mol.HasSubstructMatch(query) for mol in molecules], dtype=bool)
    return membership


def group_metrics(logits, labels, masks, rows):
    # Preserve the original float64 sigmoid and trapezoidal PR-AUC definition.
    scores = 1.0 / (1.0 + np.exp(-np.clip(np.asarray(logits, dtype=np.float64), -60., 60.)))
    roc, pr, eligible = [], [], []
    for task in range(labels.shape[1]):
        observed = rows & (masks[:, task] > 0)
        truth = labels[observed, task]
        if len(np.unique(truth)) != 2:
            continue
        value = scores[observed, task]
        roc.append(float(roc_auc_score(truth, value)))
        precision, recall, _ = precision_recall_curve(truth, value)
        pr.append(float(auc(recall, precision)))
        eligible.append(task)
    return {"molecules": int(rows.sum()), "evaluable_labels": len(eligible),
            "eligible_task_indices": eligible, "roc_auc": float(np.mean(roc)) if roc else None,
            "pr_auc": float(np.mean(pr)) if pr else None}


def validate_arrays(arrays, indices, task_names, labels, masks):
    required = {"indices", "task_names", "logits", "labels", "masks"}
    c.require(required <= set(arrays), "Prediction file lacks alignment metadata")
    c.require(np.array_equal(arrays["indices"], indices), "Prediction row order differs from the fixed test split")
    c.require(np.array_equal(arrays["task_names"], task_names), "Prediction label order differs")
    for name, expected in (("labels", labels), ("masks", masks)):
        c.require(np.array_equal(arrays[name], expected), "Prediction " + name + " differ from the dataset")
    c.require(arrays["logits"].shape == labels.shape and np.isfinite(arrays["logits"]).all(),
              "Invalid prediction shape or non-finite logits")


def aggregate(records, family, metric):
    values = [record[family][metric] for record in records]
    c.require(values and all(value is not None and np.isfinite(value) for value in values),
              "Cannot aggregate an unevaluable family")
    return {"mean": float(np.mean(values)), "std": float(np.std(values, ddof=0)), "per_seed": values}


def build_report(dataset, indices, arrays_by_seed, smoke=False):
    spec = specification()
    labels = dataset.labels[indices].numpy()
    masks = dataset.mask[indices].numpy()
    memberships = family_memberships(dataset.smiles, indices, spec["families"])
    groups = {**memberships, "overall": np.ones(len(indices), dtype=bool)}
    seeds = sorted(arrays_by_seed)
    c.require(seeds, "No completed full-model predictions found")
    records = []
    for seed in seeds:
        arrays = arrays_by_seed[seed]
        validate_arrays(arrays, indices, dataset.task_names, labels, masks)
        records.append({name: group_metrics(arrays["logits"], labels, masks, rows) for name, rows in groups.items()})
    registry, rows = [], []
    for rule in spec["families"]:
        name = rule["family"]
        stats = records[0][name]
        included = stats["molecules"] >= spec["minimum_molecules"] and stats["evaluable_labels"] >= spec["minimum_evaluable_labels"]
        registry.append({**rule, **{k: stats[k] for k in ("molecules", "evaluable_labels", "eligible_task_indices")},
                         "included": included})
    included_names = [r["family"] for r in registry if r["included"]]
    c.require(set(included_names) == set(spec["reference"]) - {"overall"}, "Chemical-family inclusion changed")
    for name in included_names + ["overall"]:
        stats, reference = records[0][name], spec["reference"][name]
        c.require(stats["molecules"] == reference["molecules"] and stats["evaluable_labels"] == reference["evaluable_labels"],
                  "Chemical-family sample or label counts changed: " + name)
        scores = {metric: aggregate(records, name, metric) for metric in ("roc_auc", "pr_auc")}
        baseline = {metric: reference[metric] for metric in ("roc_auc", "pr_auc")}
        rows.append({"family": name, "display_name": reference["display_name"],
                     "molecules": stats["molecules"], "evaluable_labels": stats["evaluable_labels"],
                     "careor": scores, "mpnn_pom": baseline,
                     "delta": {metric: scores[metric]["mean"] - baseline[metric]["mean"] for metric in scores}})
    status = "smoke_check" if smoke else "complete_four_seed_report" if seeds == sorted(c.protocol()["seeds"]) else "partial_seed_report"
    return {"status": status, "seeds": seeds, "split": "chemical_cluster",
            "baseline_mode": "published_rounded_reference", "rows": rows, "family_registry": registry,
            "test_used_for_selection": False,
            "metric_rule": "macro ROC-AUC and trapezoidal PR-AUC; observed labels with both classes; population SD",
            "per_seed_all_families": records}


def render_table(report, metric):
    label = "ROC-AUC" if metric == "roc_auc" else "PR-AUC"
    lines = ["# " + label + " across chemical families", "", "Status: " + report["status"],
             "Seeds: " + ", ".join(map(str, report["seeds"])), "",
             "CaReOR values are measured from this run. MPNN-POM uses the four-decimal manuscript reference.",
             "Mean +/- population SD; families may overlap; Overall evaluates all test molecules directly.", "",
             "| Chemical family | Molecules | Evaluable labels | CaReOR | MPNN-POM | Delta |",
             "|---|---:|---:|---:|---:|---:|"]
    for row in report["rows"]:
        a, b = row["careor"][metric], row["mpnn_pom"][metric]
        lines.append("| %s | %d | %d | %.4f +/- %.4f | %.4f +/- %.4f | %+.4f |" % (
            row["display_name"], row["molecules"], row["evaluable_labels"],
            a["mean"], a["std"], b["mean"], b["std"], row["delta"][metric]))
    return "\n".join(lines) + "\n"


def plot_report(report, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = sorted((r for r in report["rows"] if r["family"] != "overall"), key=lambda r: r["delta"]["pr_auc"], reverse=True)
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 11}):
        fig, axes = plt.subplots(1, 2, figsize=(14, 7), sharey=True)
        for axis, metric, title in zip(axes, ("roc_auc", "pr_auc"), ("ROC-AUC", "PR-AUC")):
            delta = [r["delta"][metric] for r in rows]
            bars = axis.barh(np.arange(len(rows)), delta, color=["#3579A8" if v >= 0 else "#C87362" for v in delta])
            axis.axvline(0, color="#555555", linewidth=0.8)
            axis.set_title(title)
            axis.set_xlabel("CaReOR minus MPNN-POM")
            axis.set_yticks(np.arange(len(rows)))
            axis.set_yticklabels([r["display_name"] for r in rows])
            axis.bar_label(bars, labels=["%+.4f" % v for v in delta], padding=4, fontsize=10)
            span = max(max(0, max(delta)) - min(0, min(delta)), .01)
            axis.set_xlim(min(0, min(delta)) - .28 * span, max(0, max(delta)) + .28 * span)
            axis.spines[["top", "right"]].set_visible(False)
        axes[0].invert_yaxis()
        if report["status"] != "complete_four_seed_report":
            fig.suptitle(report["status"].replace("_", " ").title())
        fig.text(.5, .015, "MPNN-POM: manuscript reference; deltas include reference rounding.", ha="center", fontsize=10)
        fig.tight_layout(rect=(0, .04, 1, .96))
        fig.savefig(output / "chemical_family_delta.png", dpi=220)
        fig.savefig(output / "chemical_family_delta.pdf")
        plt.close(fig)


def write_report(report, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    c.write(output / "chemical_summary.json", report)
    c.write(output / "family_registry.json", report["family_registry"])
    flat = []
    for row in report["rows"]:
        value = {k: row[k] for k in ("family", "display_name", "molecules", "evaluable_labels")}
        for metric in ("roc_auc", "pr_auc"):
            for model in ("careor", "mpnn_pom"):
                for stat in ("mean", "std"):
                    value[model + "_" + metric + "_" + stat] = row[model][metric][stat]
            value["delta_" + metric] = row["delta"][metric]
        flat.append(value)
    with (output / "family_comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flat[0]))
        writer.writeheader()
        writer.writerows(flat)
    for metric in ("roc_auc", "pr_auc"):
        (output / ("family_" + metric + ".md")).write_text(render_table(report, metric), encoding="utf-8")
    plot_report(report, output)


def summarize_families(result_root):
    c.require(c.SPLIT_NAME == "chemical_cluster", "Family evaluation requires the chemical-cluster partition")
    root = Path(result_root).resolve()
    c.require(root.is_dir(), "Run directory missing")
    dataset, sets, cache = c.load_dataset(build=True)
    arrays_by_seed, sources, smoke_flags = {}, [], []
    for folder in sorted(root.glob("full__seed_*")):
        config = c.read(folder / "run_config.json")
        result = c.read(folder / "result.json")
        selection = c.read(folder / "selection.json")
        seed = config["seed"]
        c.require(folder.name == "full__seed_" + str(seed) and seed not in arrays_by_seed, "Duplicate or mismatched seed")
        c.require(config["arm"] == result["arm"] == "full" and result["seed"] == seed, "Run identity mismatch")
        c.require(config["split_name"] == result["split_name"] == "chemical_cluster", "Run uses a different partition")
        c.require(config["dataset_sha256"] == c.protocol()["dataset_sha256"] and config["split"] == cache["splits"], "Run dataset or split differs")
        c.require(config["task_names"] == dataset.task_names, "Run label order differs")
        c.require(result["run_config_sha256"] == c.sha(folder / "run_config.json"), "Run configuration changed")
        c.require(result["checkpoint_sha256"] == selection["checkpoint_sha256"] == c.sha(folder / "model.pth"), "Selected checkpoint changed")
        c.require(result["selected"] == selection["selected"], "Selection record differs")
        c.require(config["smoke"] == result["smoke"], "Run scope differs")
        smoke_flags.append(config["smoke"])
        info = result["predictions"]["test"]
        c.require(info["file"] == "test_predictions.npz", "Unexpected prediction filename")
        path = folder / info["file"]
        c.require(c.sha(path) == info["sha256"], "Test predictions changed")
        with np.load(path, allow_pickle=False) as saved:
            arrays_by_seed[seed] = {key: saved[key] for key in saved.files}
        sources.append({"seed": seed, "checkpoint_sha256": result["checkpoint_sha256"], "prediction_sha256": info["sha256"]})
    c.require(smoke_flags and len(set(smoke_flags)) == 1, "Missing runs or mixed smoke/formal results")
    summary = c.read(root / "summary.json")
    c.require(sorted(r["seed"] for r in summary["results"]) == sorted(arrays_by_seed), "Incomplete run seed set")
    report = build_report(dataset, np.asarray(sets[2].indices), arrays_by_seed, smoke=smoke_flags[0])
    report.update(sources=sources, dataset_sha256=c.protocol()["dataset_sha256"],
                  split_sha256=c.protocol()["chemical_cluster_split"]["sha256"],
                  rules_sha256=c.sha(c.ROOT / "data/configures/chemical_families.json"))
    output = root / "chemical_families"
    write_report(report, output)
    print("Chemical-family report:", output, "Status:", report["status"], flush=True)
    return report
