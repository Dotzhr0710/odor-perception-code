import argparse
import math
import sys
from pathlib import Path

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

SCRIPT_DIR = Path(__file__).resolve().parent
REAL_DIR = SCRIPT_DIR.parent
ORIGIN_DIR = REAL_DIR.parent
if str(ORIGIN_DIR) not in sys.path:
    sys.path.insert(0, str(ORIGIN_DIR))

from real.data.gslf_m1_dataset import (
    apply_three_d_standardizer,
    build_gslf_m1_bundle,
    bundle_to_metadata,
    fit_three_d_standardizer,
    save_json,
    set_global_seed,
    split_bundle,
)
from real.models.m1_vnew import M1VNew, ORSpectrumClassifier
from real.tools.m1_vnew_common import (
    compute_labelwise_binary_metrics,
    compute_masked_loss,
    density_stats_from_logits,
    forward_m1_batch,
    move_batch_to_device,
    tensor_stats,
)


def str2bool(value) -> bool:
    if isinstance(value, bool):
        return value
    val = str(value).strip().lower()
    if val in {"1", "true", "yes", "y", "t"}:
        return True
    if val in {"0", "false", "no", "n", "f"}:
        return False
    raise argparse.ArgumentTypeError(f"Could not parse bool value: {value}")


def count_trainable_parameters(*modules) -> int:
    total = 0
    for module in modules:
        total += sum(param.numel() for param in module.parameters() if param.requires_grad)
    return int(total)


def split_has_3d_ratio(bundle, indices) -> float:
    if bundle.has_3d is None:
        return float("nan")
    index_tensor = torch.as_tensor(indices, dtype=torch.long)
    return float(bundle.has_3d[index_tensor].float().mean().item())


def parse_args():
    parser = argparse.ArgumentParser(description="Train M1-vNEW + GS-LF multi-label classifier.")
    parser.add_argument("--mol_csv", default="data/datasets/chemprop_gs_lf_filtered.csv")
    parser.add_argument("--morvalue_csv", default="runs/m1_vnew_inputs/morvalue.csv")
    parser.add_argument("--spectrum_pt", default="runs/gslf_or_spectrum/gslf_or_spectrum.pt")
    parser.add_argument("--out_dir", default="runs/m1_vnew_gslf")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--alpha_max", type=float, default=0.5)
    parser.add_argument("--b_max", type=float, default=0.5)
    parser.add_argument("--lambda_alpha", type=float, default=1e-3)
    parser.add_argument("--lambda_b", type=float, default=1e-3)
    parser.add_argument("--classifier_input", choices=["p1", "r1", "both"], default="p1")
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--test_ratio", type=float, default=0.1)
    parser.add_argument("--m1_mode", choices=["none", "b_only", "affine"], default="affine")
    parser.add_argument("--pos_weight", type=float, default=None)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--use_3d_branch", type=str2bool, default=False)
    parser.add_argument("--three_d_feat_path", default="")
    parser.add_argument("--three_d_hidden_dim", type=int, default=64)
    parser.add_argument("--three_d_dropout", type=float, default=0.1)
    parser.add_argument("--three_d_num_confs", type=int, default=3)
    parser.add_argument("--three_d_use_interaction_term", type=str2bool, default=True)
    parser.add_argument("--three_d_norm_type", choices=["layernorm", "batchnorm", "none"], default="layernorm")
    parser.add_argument("--three_d_fallback_zero", type=str2bool, default=True)
    parser.add_argument("--three_d_add_success_flag", type=str2bool, default=True)
    return parser.parse_args()


def evaluate_epoch(data_loader, m1_model, classifier, criterion, device, args):
    m1_model.eval()
    classifier.eval()

    total_loss = 0.0
    total_samples = 0
    y_true_all = []
    y_score_all = []
    y_mask_all = []
    alpha_all = []
    b_all = []
    r0_all = []
    r1_all = []

    with torch.no_grad():
        for batch in data_loader:
            batch = move_batch_to_device(batch, device)
            outputs = forward_m1_batch(batch, m1_model, classifier, args.m1_mode, args.classifier_input)
            labels = batch["labels"]
            label_mask = batch["label_mask"]
            loss_cls = compute_masked_loss(criterion(outputs["logits_cls"], labels), label_mask)
            loss = loss_cls
            if args.m1_mode != "none":
                loss = loss + args.lambda_alpha * ((outputs["alpha"] - 1.0) ** 2).mean()
                loss = loss + args.lambda_b * (outputs["b"] ** 2).mean()

            batch_size = labels.size(0)
            total_loss += float(loss.item()) * batch_size
            total_samples += batch_size

            y_true_all.append(labels.detach().cpu())
            y_score_all.append(torch.sigmoid(outputs["logits_cls"]).detach().cpu())
            y_mask_all.append(label_mask.detach().cpu())
            alpha_all.append(outputs["alpha"].detach().cpu())
            b_all.append(outputs["b"].detach().cpu())
            r0_all.append(outputs["r0"].detach().cpu())
            r1_all.append(outputs["r1"].detach().cpu())

    y_true = torch.cat(y_true_all).numpy()
    y_score = torch.cat(y_score_all).numpy()
    y_mask = torch.cat(y_mask_all).numpy()
    alpha = torch.cat(alpha_all)
    b = torch.cat(b_all)
    r0 = torch.cat(r0_all)
    r1 = torch.cat(r1_all)

    metrics = compute_labelwise_binary_metrics(y_true, y_score, y_mask)
    metrics["loss"] = total_loss / max(total_samples, 1)
    metrics["alpha"] = tensor_stats(alpha)
    metrics["b"] = tensor_stats(b)
    metrics["density_before"] = density_stats_from_logits(r0)
    metrics["density_after"] = density_stats_from_logits(r1)
    return metrics


def main():
    args = parse_args()
    set_global_seed(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    use_cpu = str(args.device).lower().startswith("cpu") or not torch.cuda.is_available()
    device = torch.device("cpu" if use_cpu else args.device)

    bundle = build_gslf_m1_bundle(
        args.mol_csv,
        args.morvalue_csv,
        args.spectrum_pt,
        three_d_feat_path=args.three_d_feat_path if args.use_3d_branch else None,
    )
    datasets, splits = split_bundle(bundle, val_ratio=args.val_ratio, test_ratio=args.test_ratio, seed=args.seed)

    three_d_scaler_state = None
    if args.use_3d_branch:
        if bundle.feat_3d_raw is None or bundle.has_3d is None or bundle.three_d_feature_names is None:
            raise ValueError("use_3d_branch=True but no aligned 3D cache was loaded. Please check --three_d_feat_path.")
        three_d_scaler_state = fit_three_d_standardizer(
            feat_3d_raw=bundle.feat_3d_raw,
            has_3d=bundle.has_3d,
            train_indices=splits["train"],
            feature_names=bundle.three_d_feature_names,
        )
        bundle = apply_three_d_standardizer(
            bundle,
            scaler_state=three_d_scaler_state,
            add_success_flag=args.three_d_add_success_flag,
            fallback_zero=args.three_d_fallback_zero,
        )
        save_json(three_d_scaler_state, str(out_dir / "three_d_scaler.json"))

    train_loader = DataLoader(datasets["train"], batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(datasets["val"], batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(datasets["test"], batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    num_ors = int(bundle.logits_base.shape[1])
    num_labels = int(bundle.labels.shape[1])
    classifier_input_dim = num_ors * 2 if args.classifier_input == "both" else num_ors
    three_d_input_dim = int(bundle.feat_3d.shape[1]) if bundle.feat_3d is not None else 0

    m1_model = M1VNew(alpha_max=args.alpha_max, b_max=args.b_max).to(device)
    classifier = ORSpectrumClassifier(
        input_dim=classifier_input_dim,
        num_labels=num_labels,
        use_3d_branch=args.use_3d_branch,
        three_d_input_dim=three_d_input_dim,
        three_d_hidden_dim=args.three_d_hidden_dim,
        three_d_dropout=args.three_d_dropout,
        three_d_norm_type=args.three_d_norm_type,
        three_d_use_interaction_term=args.three_d_use_interaction_term,
    ).to(device)

    params = list(classifier.parameters()) + list(m1_model.parameters())
    optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)
    pos_weight = None
    if args.pos_weight is not None:
        pos_weight = torch.full((num_labels,), float(args.pos_weight), device=device)
    criterion = nn.BCEWithLogitsLoss(reduction="none", pos_weight=pos_weight)

    config = vars(args).copy()
    config["bundle"] = bundle_to_metadata(bundle)
    config["three_d_scaler_state"] = three_d_scaler_state
    config["train_has_3d_ratio"] = split_has_3d_ratio(bundle, splits["train"])
    config["val_has_3d_ratio"] = split_has_3d_ratio(bundle, splits["val"])
    config["test_has_3d_ratio"] = split_has_3d_ratio(bundle, splits["test"])
    config["num_trainable_params"] = count_trainable_parameters(m1_model, classifier)
    save_json(config, str(out_dir / "config.json"))

    best_score = -math.inf
    history = []

    for epoch in range(1, args.epochs + 1):
        m1_model.train()
        classifier.train()
        running_loss = 0.0
        running_samples = 0

        for batch in train_loader:
            batch = move_batch_to_device(batch, device)
            outputs = forward_m1_batch(batch, m1_model, classifier, args.m1_mode, args.classifier_input)
            labels = batch["labels"]
            label_mask = batch["label_mask"]
            loss_cls = compute_masked_loss(criterion(outputs["logits_cls"], labels), label_mask)
            loss = loss_cls
            if args.m1_mode != "none":
                loss = loss + args.lambda_alpha * ((outputs["alpha"] - 1.0) ** 2).mean()
                loss = loss + args.lambda_b * (outputs["b"] ** 2).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_size = labels.size(0)
            running_loss += float(loss.item()) * batch_size
            running_samples += batch_size

        train_loss = running_loss / max(running_samples, 1)
        val_metrics = evaluate_epoch(val_loader, m1_model, classifier, criterion, device, args)
        score = val_metrics["macro_auroc"]
        if math.isnan(score):
            score = -val_metrics["loss"]

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_metrics["loss"],
            "val_macro_auroc": val_metrics["macro_auroc"],
            "val_micro_auroc": val_metrics["micro_auroc"],
            "val_macro_auprc": val_metrics["macro_auprc"],
            "val_micro_auprc": val_metrics["micro_auprc"],
            "val_mAP": val_metrics["mAP"],
            "alpha_mean": val_metrics["alpha"]["mean"],
            "alpha_std": val_metrics["alpha"]["std"],
            "alpha_min": val_metrics["alpha"]["min"],
            "alpha_max": val_metrics["alpha"]["max"],
            "b_mean": val_metrics["b"]["mean"],
            "b_std": val_metrics["b"]["std"],
            "b_min": val_metrics["b"]["min"],
            "b_max": val_metrics["b"]["max"],
            "d_soft_before": val_metrics["density_before"]["d_soft"],
            "d_soft_after": val_metrics["density_after"]["d_soft"],
            "val_has_3d_ratio": config["val_has_3d_ratio"],
        }
        history.append(row)

        print(
            f"[epoch {epoch:03d}] train_loss={train_loss:.6f} val_loss={val_metrics['loss']:.6f} "
            f"val_macro_auroc={val_metrics['macro_auroc']:.6f} val_mAP={val_metrics['mAP']:.6f}"
        )
        print(
            f"  alpha mean/std/min/max={row['alpha_mean']:.4f}/{row['alpha_std']:.4f}/"
            f"{row['alpha_min']:.4f}/{row['alpha_max']:.4f}"
        )
        print(
            f"  b mean/std/min/max={row['b_mean']:.4f}/{row['b_std']:.4f}/"
            f"{row['b_min']:.4f}/{row['b_max']:.4f}"
        )
        print(
            f"  d_soft before/after={row['d_soft_before']:.6f}/{row['d_soft_after']:.6f}"
        )
        if args.use_3d_branch:
            print(
                f"  has_3d ratio train/val/test="
                f"{config['train_has_3d_ratio']:.4f}/{config['val_has_3d_ratio']:.4f}/{config['test_has_3d_ratio']:.4f}"
            )

        state = {
            "epoch": epoch,
            "m1_state_dict": m1_model.state_dict(),
            "classifier_state_dict": classifier.state_dict(),
            "label_names": bundle.label_names,
            "seq_ids": bundle.seq_ids,
            "splits": {name: indices.tolist() for name, indices in splits.items()},
            "config": config,
            "val_metrics": val_metrics,
            "test_size": len(datasets["test"]),
            "three_d_feature_names": bundle.three_d_feature_names,
            "three_d_model_feature_names": bundle.three_d_model_feature_names,
        }
        torch.save(state, out_dir / "last_model.pt")
        if score > best_score:
            best_score = score
            torch.save(state, out_dir / "best_model.pt")

    pd.DataFrame(history).to_csv(out_dir / "train_log.csv", index=False, encoding="utf-8")

    best_checkpoint = torch.load(out_dir / "best_model.pt", map_location="cpu")
    m1_model.load_state_dict(best_checkpoint["m1_state_dict"])
    classifier.load_state_dict(best_checkpoint["classifier_state_dict"])
    test_metrics = evaluate_epoch(test_loader, m1_model, classifier, criterion, device, args)
    save_json(test_metrics, str(out_dir / "test_metrics_from_best.json"))
    print(
        f"[done] best val score={best_score:.6f}, test macro AUROC={test_metrics['macro_auroc']:.6f}, "
        f"test mAP={test_metrics['mAP']:.6f}, params={config['num_trainable_params']}"
    )


if __name__ == "__main__":
    main()
