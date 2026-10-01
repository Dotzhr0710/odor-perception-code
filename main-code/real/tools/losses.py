"""Masked task objectives used by the submitted CaReOR recipe."""
from __future__ import annotations
import math
from typing import Dict, Mapping, Tuple
import torch
import torch.nn.functional as F

def _require_aligned(
    logits: torch.Tensor, labels: torch.Tensor, masks: torch.Tensor
) -> None:
    if not all(isinstance(value, torch.Tensor) for value in (logits, labels, masks)):
        raise TypeError("logits, labels, and masks must be torch tensors")
    if logits.ndim != 2 or logits.numel() == 0:
        raise ValueError("logits must be a non-empty rank-2 tensor")
    if logits.shape != labels.shape or logits.shape != masks.shape:
        raise ValueError("logits, labels, and masks must have identical shapes")
    if not torch.is_floating_point(logits):
        raise ValueError("logits must be floating point")
    if logits.device != labels.device or logits.device != masks.device:
        raise ValueError("logits, labels, and masks must share a device")
    if not bool(torch.isfinite(logits).all()):
        raise ValueError("logits must be finite")
    valid = masks > 0
    if bool(valid.any()):
        observed = labels[valid]
        if not bool(torch.isfinite(observed).all()):
            raise ValueError("observed labels must be finite")
        if not bool(((observed == 0.0) | (observed == 1.0)).all()):
            raise ValueError("observed labels must be binary")

def _finite_nonnegative(name: str, value: float) -> float:
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError("{} must be finite and non-negative".format(name))
    return result

def _finite_positive(name: str, value: float) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError("{} must be finite and positive".format(name))
    return result

def build_train_label_statistics(
    labels: torch.Tensor,
    masks: torch.Tensor,
    smoothing: float = 0.5,
    maximum_positive_weight: float = 8.0,
) -> Dict[str, torch.Tensor]:
    """Build immutable class statistics from the training split only.

    The square-root ratio is intentionally milder than inverse prevalence.
    Tasks that do not contain both classes remain part of the micro and macro
    BCE terms, but are excluded from class-balanced and pairwise components.
    """

    if not isinstance(labels, torch.Tensor) or not isinstance(masks, torch.Tensor):
        raise TypeError("labels and masks must be torch tensors")
    if labels.ndim != 2 or labels.shape != masks.shape or labels.numel() == 0:
        raise ValueError("labels and masks must be aligned non-empty matrices")
    smoothing = _finite_positive("smoothing", smoothing)
    maximum_positive_weight = _finite_positive(
        "maximum_positive_weight", maximum_positive_weight
    )
    valid = masks > 0
    observed = labels[valid]
    if observed.numel() and (
        not bool(torch.isfinite(observed).all())
        or not bool(((observed == 0.0) | (observed == 1.0)).all())
    ):
        raise ValueError("observed training labels must be finite and binary")
    safe_labels = torch.where(valid, labels, torch.zeros_like(labels))
    positive_counts = ((safe_labels > 0.5) & valid).sum(dim=0).to(torch.float64)
    observed_counts = valid.sum(dim=0).to(torch.float64)
    negative_counts = observed_counts - positive_counts
    eligible = (positive_counts > 0) & (negative_counts > 0)
    positive_weights = torch.sqrt(
        (negative_counts + smoothing) / (positive_counts + smoothing)
    ).clamp(min=1.0, max=maximum_positive_weight)
    return {
        "observed_counts": observed_counts.cpu(),
        "positive_counts": positive_counts.cpu(),
        "negative_counts": negative_counts.cpu(),
        "positive_weights": positive_weights.cpu(),
        "eligible_tasks": eligible.cpu(),
    }

def masked_micro_bce(
    logits: torch.Tensor, labels: torch.Tensor, masks: torch.Tensor
) -> torch.Tensor:
    """Observed-label BCE averaged over every observed molecule/task pair."""

    _require_aligned(logits, labels, masks)
    valid = masks > 0
    if not bool(valid.any()):
        return logits.sum() * 0.0
    safe_labels = torch.where(valid, labels, torch.zeros_like(labels))
    losses = F.binary_cross_entropy_with_logits(logits, safe_labels, reduction="none")
    return (losses * valid.to(losses.dtype)).sum() / valid.sum().clamp_min(1)

def masked_macro_bce(
    logits: torch.Tensor, labels: torch.Tensor, masks: torch.Tensor
) -> torch.Tensor:
    """Observed-label BCE with equal weight for each represented odor task."""

    _require_aligned(logits, labels, masks)
    valid = masks > 0
    safe_labels = torch.where(valid, labels, torch.zeros_like(labels))
    losses = F.binary_cross_entropy_with_logits(logits, safe_labels, reduction="none")
    counts = valid.sum(dim=0)
    represented = counts > 0
    if not bool(represented.any()):
        return logits.sum() * 0.0
    per_task = (losses * valid.to(losses.dtype)).sum(dim=0) / counts.clamp_min(1)
    return per_task[represented].mean()

def masked_balanced_macro_bce(
    logits: torch.Tensor,
    labels: torch.Tensor,
    masks: torch.Tensor,
    label_statistics: Mapping[str, torch.Tensor],
) -> torch.Tensor:
    """Macro BCE with train-only, capped positive-class reweighting."""

    _require_aligned(logits, labels, masks)
    if "positive_weights" not in label_statistics or "eligible_tasks" not in label_statistics:
        raise ValueError("label statistics lack positive_weights or eligible_tasks")
    positive_weights = label_statistics["positive_weights"].to(
        device=logits.device, dtype=logits.dtype
    )
    eligible = label_statistics["eligible_tasks"].to(device=logits.device).bool()
    if tuple(positive_weights.shape) != (logits.size(1),) or tuple(eligible.shape) != (
        logits.size(1),
    ):
        raise ValueError("label statistics task dimension drift")
    valid = masks > 0
    safe_labels = torch.where(valid, labels, torch.zeros_like(labels))
    losses = F.binary_cross_entropy_with_logits(logits, safe_labels, reduction="none")
    sample_weights = torch.where(
        safe_labels > 0.5,
        positive_weights.unsqueeze(0).expand_as(logits),
        torch.ones_like(logits),
    )
    weighted_valid = sample_weights * valid.to(logits.dtype)
    denominators = weighted_valid.sum(dim=0)
    represented = eligible & (denominators > 0)
    if not bool(represented.any()):
        return logits.sum() * 0.0
    per_task = (losses * weighted_valid).sum(dim=0) / denominators.clamp_min(1.0e-12)
    return per_task[represented].mean()

def masked_pairwise_auc_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    masks: torch.Tensor,
    label_statistics: Mapping[str, torch.Tensor],
    temperature: float = 1.0,
) -> torch.Tensor:
    """Deterministic per-task positive/negative ranking surrogate.

    The loss is divided by log(2), so tied scores have unit loss.  Every
    positive/negative pair present in the current minibatch is used; there is
    no stochastic pair sampler that could break paired-profile reproducibility.
    """

    _require_aligned(logits, labels, masks)
    temperature = _finite_positive("temperature", temperature)
    eligible = label_statistics.get("eligible_tasks")
    if not isinstance(eligible, torch.Tensor) or tuple(eligible.shape) != (
        logits.size(1),
    ):
        raise ValueError("label statistics eligible_tasks dimension drift")
    eligible = eligible.to(device=logits.device).bool()
    valid = masks > 0
    positive = valid & (labels > 0.5)
    negative = valid & (labels <= 0.5)
    pair_mask = (
        positive[:, None, :]
        & negative[None, :, :]
        & eligible[None, None, :]
    )
    pair_counts = pair_mask.sum(dim=(0, 1))
    represented = pair_counts > 0
    if not bool(represented.any()):
        return logits.sum() * 0.0
    margins = (logits[:, None, :] - logits[None, :, :]) / temperature
    losses = F.softplus(-margins) / math.log(2.0)
    per_task = (
        losses * pair_mask.to(losses.dtype)
    ).sum(dim=(0, 1)) / pair_counts.clamp_min(1).to(losses.dtype)
    return per_task[represented].mean()

def direct_task_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    masks: torch.Tensor,
    label_statistics: Mapping[str, torch.Tensor],
    profile: Mapping[str, float],
    pairwise_temperature: float = 1.0,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Combine registered direct task losses and return every component."""

    expected = {
        "micro_bce_weight",
        "macro_bce_weight",
        "balanced_macro_bce_weight",
        "pairwise_auc_weight",
    }
    if set(profile.keys()) != expected:
        raise ValueError("direct loss profile keys drift")
    weights = {name: _finite_nonnegative(name, profile[name]) for name in expected}
    if abs(sum(weights.values()) - 1.0) > 1.0e-8:
        raise ValueError("direct loss profile weights must sum to one")
    components = {
        "micro_bce": masked_micro_bce(logits, labels, masks),
        "macro_bce": masked_macro_bce(logits, labels, masks),
    }
    if weights["balanced_macro_bce_weight"] > 0.0:
        components["balanced_macro_bce"] = masked_balanced_macro_bce(
            logits, labels, masks, label_statistics
        )
    else:
        components["balanced_macro_bce"] = logits.sum() * 0.0
    if weights["pairwise_auc_weight"] > 0.0:
        components["pairwise_auc"] = masked_pairwise_auc_loss(
            logits,
            labels,
            masks,
            label_statistics,
            temperature=pairwise_temperature,
        )
    else:
        components["pairwise_auc"] = logits.sum() * 0.0
    total = (
        weights["micro_bce_weight"] * components["micro_bce"]
        + weights["macro_bce_weight"] * components["macro_bce"]
        + weights["balanced_macro_bce_weight"]
        * components["balanced_macro_bce"]
        + weights["pairwise_auc_weight"] * components["pairwise_auc"]
    )
    return total, components
