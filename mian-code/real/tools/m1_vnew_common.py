import math
from typing import Dict, List

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

from real.models.m1_vnew import apply_m1_calibration, build_classifier_input, summarize_spectrum_logits


def move_batch_to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    result = {}
    for key, value in batch.items():
        result[key] = value.to(device) if torch.is_tensor(value) else value
    return result


def compute_labelwise_binary_metrics(y_true: np.ndarray, y_score: np.ndarray, y_mask: np.ndarray) -> Dict[str, float]:
    metrics = {"macro_auroc": float("nan"), "micro_auroc": float("nan"), "macro_auprc": float("nan"),
               "micro_auprc": float("nan"), "mAP": float("nan"), "valid_label_count": 0}

    valid_aurocs: List[float] = []
    valid_auprcs: List[float] = []
    for idx in range(y_true.shape[1]):
        valid_mask = y_mask[:, idx] > 0
        if valid_mask.sum() == 0:
            continue
        label_true = y_true[valid_mask, idx]
        label_score = y_score[valid_mask, idx]
        unique = np.unique(label_true)
        if unique.shape[0] < 2:
            continue
        valid_aurocs.append(roc_auc_score(label_true, label_score))
        valid_auprcs.append(average_precision_score(label_true, label_score))

    if valid_aurocs:
        metrics["macro_auroc"] = float(np.mean(valid_aurocs))
        metrics["valid_label_count"] = len(valid_aurocs)
    else:
        print("[warning] macro AUROC undefined because no label has both classes in this split.")

    if valid_auprcs:
        metrics["macro_auprc"] = float(np.mean(valid_auprcs))
        metrics["mAP"] = float(np.mean(valid_auprcs))
    else:
        print("[warning] macro AUPRC undefined because no label has valid positives in this split.")

    flat_mask = y_mask.reshape(-1) > 0
    flat_true = y_true.reshape(-1)[flat_mask]
    flat_score = y_score.reshape(-1)[flat_mask]
    if np.unique(flat_true).shape[0] >= 2:
        metrics["micro_auroc"] = float(roc_auc_score(flat_true, flat_score))
        metrics["micro_auprc"] = float(average_precision_score(flat_true, flat_score))
    else:
        print("[warning] micro metrics undefined because flattened labels are single-class.")

    return metrics


def compute_masked_loss(loss_matrix: torch.Tensor, label_mask: torch.Tensor) -> torch.Tensor:
    valid_count = label_mask.sum().clamp(min=1.0)
    return (loss_matrix * label_mask).sum() / valid_count


def forward_m1_batch(
    batch: Dict[str, torch.Tensor],
    m1_model,
    classifier,
    m1_mode: str,
    classifier_input_mode: str,
):
    r0 = batch["logits_base"]
    summary = summarize_spectrum_logits(r0)
    m1_out = m1_model(batch["rdkit_z"], summary)
    calibrated = apply_m1_calibration(r0, m1_out["alpha"], m1_out["b"], mode=m1_mode)
    cls_input = build_classifier_input(calibrated["r1"], calibrated["p1"], classifier_input_mode)
    logits_cls = classifier(cls_input, batch.get("feat_3d"))
    return {
        "logits_cls": logits_cls,
        "summary": summary,
        "r0": r0,
        "r1": calibrated["r1"],
        "p1": calibrated["p1"],
        "alpha": calibrated["alpha"],
        "b": calibrated["b"],
    }


def tensor_stats(tensor: torch.Tensor) -> Dict[str, float]:
    return {
        "mean": float(tensor.mean().item()),
        "std": float(tensor.std(unbiased=False).item()),
        "min": float(tensor.min().item()),
        "max": float(tensor.max().item()),
    }


def density_stats_from_logits(logits: torch.Tensor) -> Dict[str, float]:
    probs = torch.sigmoid(logits)
    return {
        "d_soft": float(probs.mean().item()),
        "d_hard": float((probs > 0.5).float().mean().item()),
        "near_threshold": float(((probs >= 0.4) & (probs <= 0.6)).float().mean().item()),
    }


def safe_float(value) -> float:
    if isinstance(value, float):
        return value
    if hasattr(value, "item"):
        return float(value.item())
    return float(value)


def maybe_nanmean(values: List[float]) -> float:
    finite = [value for value in values if not math.isnan(value)]
    return float(np.mean(finite)) if finite else float("nan")
