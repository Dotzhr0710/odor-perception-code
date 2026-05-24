import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

SCRIPT_DIR = Path(__file__).resolve().parent
REAL_DIR = SCRIPT_DIR.parent
ORIGIN_DIR = REAL_DIR.parent
if str(ORIGIN_DIR) not in sys.path:
    sys.path.insert(0, str(ORIGIN_DIR))

from real.data.gslf_m1_dataset import (
    GSLFM1Dataset,
    apply_three_d_standardizer,
    build_gslf_m1_bundle,
    save_json,
)
from real.models.m1_vnew import M1VNew, ORSpectrumClassifier
from real.tools.m1_vnew_common import (
    compute_labelwise_binary_metrics,
    forward_m1_batch,
    move_batch_to_device,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate M1-vNEW on GS-LF test split.")
    parser.add_argument("--model_pt", required=True)
    parser.add_argument("--mol_csv", default="data/datasets/chemprop_gs_lf_filtered.csv")
    parser.add_argument("--morvalue_csv", default="runs/m1_vnew_inputs/morvalue.csv")
    parser.add_argument("--spectrum_pt", default="runs/gslf_or_spectrum/gslf_or_spectrum.pt")
    parser.add_argument("--out_dir", default="runs/m1_vnew_gslf_eval")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=0)
    return parser.parse_args()


def per_sample_density(probs: torch.Tensor):
    return {
        "d_soft": probs.mean(dim=1),
        "d_hard": (probs > 0.5).float().mean(dim=1),
        "near_threshold": ((probs >= 0.4) & (probs <= 0.6)).float().mean(dim=1),
    }


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = torch.load(args.model_pt, map_location="cpu")
    config = checkpoint["config"]
    bundle = build_gslf_m1_bundle(
        args.mol_csv,
        args.morvalue_csv,
        args.spectrum_pt,
        three_d_feat_path=config.get("three_d_feat_path") if config.get("use_3d_branch") else None,
    )
    if config.get("use_3d_branch"):
        bundle = apply_three_d_standardizer(
            bundle,
            scaler_state=config["three_d_scaler_state"],
            add_success_flag=bool(config.get("three_d_add_success_flag", False)),
            fallback_zero=bool(config.get("three_d_fallback_zero", True)),
        )

    splits = checkpoint["splits"]
    test_indices = splits["test"]
    test_dataset = GSLFM1Dataset(bundle, test_indices)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    num_ors = int(bundle.logits_base.shape[1])
    classifier_input_dim = num_ors * 2 if config["classifier_input"] == "both" else num_ors
    num_labels = len(checkpoint["label_names"])
    three_d_input_dim = int(bundle.feat_3d.shape[1]) if bundle.feat_3d is not None else 0
    use_cpu = str(args.device).lower().startswith("cpu") or not torch.cuda.is_available()
    device = torch.device("cpu" if use_cpu else args.device)

    m1_model = M1VNew(alpha_max=config["alpha_max"], b_max=config["b_max"]).to(device)
    classifier = ORSpectrumClassifier(
        input_dim=classifier_input_dim,
        num_labels=num_labels,
        use_3d_branch=bool(config.get("use_3d_branch", False)),
        three_d_input_dim=three_d_input_dim,
        three_d_hidden_dim=int(config.get("three_d_hidden_dim", 64)),
        three_d_dropout=float(config.get("three_d_dropout", 0.1)),
        three_d_norm_type=config.get("three_d_norm_type", "layernorm"),
        three_d_use_interaction_term=bool(config.get("three_d_use_interaction_term", True)),
    ).to(device)
    m1_model.load_state_dict(checkpoint["m1_state_dict"])
    classifier.load_state_dict(checkpoint["classifier_state_dict"])
    m1_model.eval()
    classifier.eval()

    y_true_all = []
    y_score_all = []
    y_mask_all = []
    logits_pred_all = []
    r0_all = []
    r1_all = []
    p1_all = []
    alpha_all = []
    b_all = []
    smiles_all = []

    with torch.no_grad():
        for batch in test_loader:
            smiles_all.extend(batch["smiles"])
            batch = move_batch_to_device(batch, device)
            outputs = forward_m1_batch(batch, m1_model, classifier, config["m1_mode"], config["classifier_input"])
            y_true_all.append(batch["labels"].detach().cpu())
            y_score_all.append(torch.sigmoid(outputs["logits_cls"]).detach().cpu())
            y_mask_all.append(batch["label_mask"].detach().cpu())
            logits_pred_all.append(outputs["logits_cls"].detach().cpu())
            r0_all.append(outputs["r0"].detach().cpu())
            r1_all.append(outputs["r1"].detach().cpu())
            p1_all.append(outputs["p1"].detach().cpu())
            alpha_all.append(outputs["alpha"].detach().cpu())
            b_all.append(outputs["b"].detach().cpu())

    y_true = torch.cat(y_true_all)
    y_score = torch.cat(y_score_all)
    y_mask = torch.cat(y_mask_all)
    logits_pred = torch.cat(logits_pred_all)
    r0 = torch.cat(r0_all)
    r1 = torch.cat(r1_all)
    p1 = torch.cat(p1_all)
    alpha = torch.cat(alpha_all).squeeze(1)
    b = torch.cat(b_all).squeeze(1)

    metrics = compute_labelwise_binary_metrics(y_true.numpy(), y_score.numpy(), y_mask.numpy())
    metrics["test_has_3d_ratio"] = float(bundle.has_3d[test_indices].float().mean().item()) if bundle.has_3d is not None else float("nan")
    metrics["three_d_valid_ratio_full"] = (
        float(bundle.three_d_feature_stats["valid_3d_ratio"]) if bundle.three_d_feature_stats is not None else float("nan")
    )
    save_json(metrics, str(out_dir / "classification_metrics.json"))

    probs_base = torch.sigmoid(r0)
    density_base = per_sample_density(probs_base)
    density_new = per_sample_density(p1)
    summary_df = pd.DataFrame(
        {
            "smiles": smiles_all,
            "alpha": alpha.numpy(),
            "b": b.numpy(),
            "mean_logit_base": r0.mean(dim=1).numpy(),
            "mean_logit_new": r1.mean(dim=1).numpy(),
            "d_soft_base": density_base["d_soft"].numpy(),
            "d_soft_new": density_new["d_soft"].numpy(),
            "d_hard_base": density_base["d_hard"].numpy(),
            "d_hard_new": density_new["d_hard"].numpy(),
            "near_threshold_base": density_base["near_threshold"].numpy(),
            "near_threshold_new": density_new["near_threshold"].numpy(),
            "has_3d": bundle.has_3d[test_indices].squeeze(1).numpy() if bundle.has_3d is not None else np.nan,
        }
    )
    summary_df.to_csv(out_dir / "m1_vnew_spectrum_summary.csv", index=False, encoding="utf-8")

    torch.save(
        {
            "smiles": smiles_all,
            "r0": r0,
            "r1": r1,
            "p1": p1,
            "y_true": y_true,
            "y_mask": y_mask,
            "y_pred_logits": logits_pred,
            "label_names": checkpoint["label_names"],
            "seq_ids": checkpoint.get("seq_ids", []),
        },
        out_dir / "test_predictions.pt",
    )

    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(f"[done] wrote evaluation outputs to {out_dir}")


if __name__ == "__main__":
    main()
