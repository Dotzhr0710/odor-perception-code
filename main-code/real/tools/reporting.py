"""Per-seed metrics and aggregate result tables for CaReOR."""
import json
import statistics
from pathlib import Path


def summarize(results, output):
    output = Path(output)
    groups = {}
    for item in results:
        groups.setdefault(item["arm"], []).append(item)
    lines = ["# CaReOR results", "", "Smoke checks are not experimental evidence." if any(x["smoke"] for x in results) else "Validation-ROC selected checkpoints; mean +/- population SD.", "", "| Configuration | Test ROC-AUC | Test PR-AUC |", "|---|---:|---:|"]
    summary = {}
    for arm, rows in groups.items():
        summary[arm] = {}
        cells = []
        for split in ("val", "test"):
            for metric in ("roc_auc", "pr_auc"):
                values = [row["metrics"][split][metric] for row in rows]
                entry = {"mean": statistics.mean(values), "std": statistics.pstdev(values), "per_seed": values}
                summary[arm][split + "_" + metric] = entry
                if split == "test":
                    cells.append("%.6f +/- %.6f" % (entry["mean"], entry["std"]))
        lines.append("| %s | %s | %s |" % (arm, *cells))
    (output / "summary.json").write_text(json.dumps({"results": results, "summary": summary}, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
