"""The full-model masked BCE objective, preserving each recorded operation order."""
import torch
import torch.nn.functional as F
from real.tools import runtime as c


def staged_task_loss(logits, labels, masks, profile, split_name):
    """Match the original random and chemical-cluster backward computations."""
    c.require(logits.ndim == 2 and logits.numel() > 0 and logits.shape == labels.shape == masks.shape,
              "Loss inputs must be aligned nonempty matrices")
    c.require(logits.device == labels.device == masks.device and torch.isfinite(logits).all().item(),
              "Loss inputs must share a device and predictions must be finite")
    c.require(profile == {"micro_bce_weight": .55, "macro_bce_weight": .45,
                          "balanced_macro_bce_weight": 0, "pairwise_auc_weight": 0},
              "Full-model loss configuration changed")
    observed = masks > 0
    truth = labels[observed]
    c.require(torch.isfinite(truth).all().item() and ((truth == 0) | (truth == 1)).all().item(),
              "Observed training labels must be binary and finite")
    safe_labels = torch.where(observed, labels, torch.zeros_like(labels))
    counts = observed.sum(0)
    represented = counts > 0
    if split_name == "random":
        # The main experiment shares one BCE tensor between its two reductions.
        elements = F.binary_cross_entropy_with_logits(logits, safe_labels, reduction="none") * observed
        micro = elements.sum() / observed.sum().clamp(min=1)
        macro = ((elements.sum(0) / counts.clamp(min=1))[represented].mean()
                 if represented.any() else logits.sum() * 0)
        return profile["micro_bce_weight"] * micro + profile["macro_bce_weight"] * macro
    c.require(split_name == "chemical_cluster", "Unknown full-model loss route")
    # The chemical-cluster experiment evaluates the two BCE terms separately.
    # Keep this backward graph and the original zero-valued terms unchanged.
    losses = F.binary_cross_entropy_with_logits(logits, safe_labels, reduction="none")
    micro = ((losses * observed.to(losses.dtype)).sum() / observed.sum().clamp_min(1)
             if observed.any() else logits.sum() * 0.0)
    losses = F.binary_cross_entropy_with_logits(logits, safe_labels, reduction="none")
    macro = (((losses * observed.to(losses.dtype)).sum(0) / counts.clamp_min(1))[represented].mean()
             if represented.any() else logits.sum() * 0.0)
    balanced_zero = logits.sum() * 0.0
    pairwise_zero = logits.sum() * 0.0
    return (profile["micro_bce_weight"] * micro + profile["macro_bce_weight"] * macro
            + profile["balanced_macro_bce_weight"] * balanced_zero
            + profile["pairwise_auc_weight"] * pairwise_zero)
