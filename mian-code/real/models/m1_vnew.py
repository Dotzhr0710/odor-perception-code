


import torch
from torch import nn


def summarize_spectrum_logits(r0: torch.Tensor) -> torch.Tensor:

    if r0.ndim != 2:
        raise ValueError(f"Expected r0 to have shape [B, K], got {tuple(r0.shape)}")

    if r0.size(1) == 0:
        raise ValueError("Expected at least one OR dimension in r0.")

    mean_r = r0.mean(dim=1)
    std_r = r0.std(dim=1, unbiased=False)
    quantiles = torch.quantile(r0, q=torch.tensor([0.1, 0.5, 0.9], device=r0.device), dim=1)
    q10, q50, q90 = quantiles[0], quantiles[1], quantiles[2]

    top1 = torch.topk(r0, k=1, dim=1).values.mean(dim=1)
    top5_k = min(5, r0.size(1))
    top10_k = min(10, r0.size(1))
    top5_mean = torch.topk(r0, k=top5_k, dim=1).values.mean(dim=1)
    top10_mean = torch.topk(r0, k=top10_k, dim=1).values.mean(dim=1)

    # Map OR logits [B, R] to compact spectrum descriptors [B, 8].
    return torch.stack(
        [mean_r, std_r, q10, q50, q90, top1, top5_mean, top10_mean],
        dim=1,
    ).to(dtype=r0.dtype)


def build_norm_1d(norm_type: str, hidden_dim: int) -> nn.Module:

    norm_key = str(norm_type).strip().lower()
    if norm_key == "layernorm":
        return nn.LayerNorm(hidden_dim)
    if norm_key == "batchnorm":
        return nn.BatchNorm1d(hidden_dim)
    if norm_key == "none":
        return nn.Identity()
    raise ValueError(f"Unsupported norm_type: {norm_type}")


class ThreeDFeatureEncoder(nn.Module):


    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        output_dim: int = 64,
        dropout: float = 0.1,
        norm_type: str = "layernorm",
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            build_norm_1d(norm_type, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            build_norm_1d(norm_type, output_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, feat_3d: torch.Tensor) -> torch.Tensor:
        if feat_3d.ndim != 2:
            raise ValueError(f"Expected feat_3d to have shape [B, D], got {tuple(feat_3d.shape)}")
        return self.net(feat_3d)


class M1GeometryCompensator(nn.Module):


    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        dropout: float = 0.1,
        norm_type: str = "layernorm",
    ):
        super().__init__()
        if input_dim <= 0:
            raise ValueError("input_dim must be positive for M1GeometryCompensator.")

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            build_norm_1d(norm_type, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            build_norm_1d(norm_type, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, feat_3d: torch.Tensor) -> torch.Tensor:
        if feat_3d.ndim != 2:
            raise ValueError(f"Expected feat_3d to have shape [B, D], got {tuple(feat_3d.shape)}")

        return self.net(feat_3d)


class ConformerGeometryEncoder(nn.Module):


    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        dropout: float = 0.1,
        norm_type: str = "layernorm",
    ):
        super().__init__()
        if input_dim <= 0:
            raise ValueError("input_dim must be positive for ConformerGeometryEncoder.")
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            build_norm_1d(norm_type, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            build_norm_1d(norm_type, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, conf_feat: torch.Tensor) -> torch.Tensor:
        if conf_feat.ndim != 3:
            raise ValueError(f"Expected conf_feat to have shape [B, K, D], got {tuple(conf_feat.shape)}")
        batch_size, max_confs, input_dim = conf_feat.shape
        if input_dim != self.input_dim:
            raise ValueError(f"Expected conformer input dim {self.input_dim}, got {input_dim}")
        flat = conf_feat.reshape(batch_size * max_confs, input_dim)
        encoded = self.net(flat)
        return encoded.reshape(batch_size, max_confs, self.hidden_dim)


class ConformerScoreHead(nn.Module):


    def __init__(
        self,
        hidden_dim: int = 64,
        use_energy: bool = True,
        dropout: float = 0.1,
        context_dim: int = 0,
    ):
        super().__init__()
        self.use_energy = bool(use_energy)
        self.context_dim = int(context_dim)
        score_input_dim = int(hidden_dim) + (2 if self.use_energy else 0) + self.context_dim
        self.net = nn.Sequential(
            nn.Linear(score_input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        conf_z: torch.Tensor,
        conf_energy: torch.Tensor = None,
        conf_delta_energy: torch.Tensor = None,
        context_features: torch.Tensor = None,
    ) -> torch.Tensor:
        if conf_z.ndim != 3:
            raise ValueError(f"Expected conf_z to have shape [B, K, H], got {tuple(conf_z.shape)}")
        score_parts = [conf_z]
        if self.use_energy:
            if conf_energy is None or conf_delta_energy is None:
                raise ValueError("conf_energy and conf_delta_energy are required when use_energy=True.")
            if conf_energy.ndim == 2:
                conf_energy = conf_energy.unsqueeze(-1)
            if conf_delta_energy.ndim == 2:
                conf_delta_energy = conf_delta_energy.unsqueeze(-1)
            score_parts.extend([conf_energy, conf_delta_energy])
        if self.context_dim > 0:
            if context_features is None:
                raise ValueError("context_features are required when context_dim > 0.")
            if context_features.ndim != 2 or context_features.size(0) != conf_z.size(0):
                raise ValueError(
                    f"Expected context_features to have shape [B, C], got {tuple(context_features.shape)}"
                )
            if context_features.size(1) != self.context_dim:
                raise ValueError(f"Expected context dim {self.context_dim}, got {context_features.size(1)}")

            score_parts.append(context_features.unsqueeze(1).expand(-1, conf_z.size(1), -1))
        score_input = torch.cat(score_parts, dim=-1)
        return self.net(score_input).squeeze(-1)


class M1GeometryCompensatorConformer(nn.Module):


    def __init__(
        self,
        conf_input_dim: int,
        hidden_dim: int = 64,
        dropout: float = 0.1,
        norm_type: str = "layernorm",
        use_energy: bool = True,
        attention_mode: str = "unconditional",
        context_dim: int = 0,
    ):
        super().__init__()
        self.use_energy = bool(use_energy)
        self.attention_mode = str(attention_mode)
        if self.attention_mode not in {"unconditional", "conditional"}:
            raise ValueError(f"Unsupported attention_mode: {attention_mode}")
        self.context_dim = int(context_dim) if self.attention_mode == "conditional" else 0
        self.encoder = ConformerGeometryEncoder(
            input_dim=conf_input_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
            norm_type=norm_type,
        )
        self.score_head = ConformerScoreHead(
            hidden_dim=hidden_dim,
            use_energy=use_energy,
            dropout=dropout,
            context_dim=self.context_dim,
        )
        self.delta_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        conf_feat: torch.Tensor,
        conf_energy: torch.Tensor = None,
        conf_delta_energy: torch.Tensor = None,
        conf_mask: torch.Tensor = None,
        context_features: torch.Tensor = None,
    ):
        if conf_feat.ndim != 3:
            raise ValueError(f"Expected conf_feat to have shape [B, K, D], got {tuple(conf_feat.shape)}")
        batch_size, max_confs, _ = conf_feat.shape
        if conf_mask is None:
            conf_mask = torch.ones((batch_size, max_confs), dtype=conf_feat.dtype, device=conf_feat.device)
        conf_mask = conf_mask.to(device=conf_feat.device, dtype=conf_feat.dtype)
        valid = conf_mask > 0.5
        valid_counts = valid.sum(dim=1)


        conf_z = self.encoder(conf_feat)
        if self.attention_mode == "conditional" and context_features is None:
            raise ValueError("conditional conformer attention requires context_features.")

        scores = self.score_head(
            conf_z,
            conf_energy,
            conf_delta_energy,
            context_features=context_features,
        )
        masked_scores = scores.masked_fill(~valid, -1.0e9)


        weights = torch.softmax(masked_scores, dim=1)
        weights = torch.where(valid_counts.unsqueeze(1) > 0, weights, torch.zeros_like(weights))
        weights = weights * conf_mask
        weights = weights / weights.sum(dim=1, keepdim=True).clamp(min=1.0e-8)


        z_3d = torch.sum(weights.unsqueeze(-1) * conf_z, dim=1)
        delta_b_3d = self.delta_head(z_3d)

        has_valid = (valid_counts > 0).to(dtype=delta_b_3d.dtype, device=delta_b_3d.device).unsqueeze(1)
        delta_b_3d = delta_b_3d * has_valid

        entropy_per_sample = -(weights * torch.log(weights.clamp_min(1.0e-8))).sum(dim=1)
        valid_entropy = torch.where(valid_counts > 0, entropy_per_sample, torch.zeros_like(entropy_per_sample))
        denom = (valid_counts > 0).to(dtype=conf_feat.dtype).sum().clamp(min=1.0)
        conf_entropy = valid_entropy.sum() / denom

        return {
            "delta_b_3d": delta_b_3d,
            "z_3d": z_3d,
            "conf_score_logits": scores,
            "conf_attn_weights": weights,
            "conf_valid_counts": valid_counts,
            "conf_entropy": conf_entropy,
            "attention_context_dim": self.context_dim,
        }


class SpectrumResidualReEncoder(nn.Module):


    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        dropout: float = 0.1,
        norm_type: str = "layernorm",
        residual_scale: float = 0.1,
        residual_space: str = "logit",
        output_dim: int = None,
    ):
        super().__init__()
        if input_dim <= 0:
            raise ValueError("input_dim must be positive for SpectrumResidualReEncoder.")
        if residual_space not in {"logit", "prob"}:
            raise ValueError(f"Unsupported residual_space: {residual_space}")
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim) if output_dim is not None else int(input_dim)
        if self.output_dim <= 0:
            raise ValueError("output_dim must be positive for SpectrumResidualReEncoder.")
        self.residual_scale = float(residual_scale)
        self.residual_space = str(residual_space)
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            build_norm_1d(norm_type, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            build_norm_1d(norm_type, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, self.output_dim),
        )

        # Start M3 as an identity refinement so training begins from the M1 spectrum.
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, spectrum_features: torch.Tensor, spectrum_logits: torch.Tensor = None):
        if spectrum_features.ndim != 2:
            raise ValueError(
                f"Expected spectrum_features to have shape [B, R], got {tuple(spectrum_features.shape)}"
            )
        if spectrum_features.size(1) != self.input_dim:
            raise ValueError(f"Expected spectrum input dim {self.input_dim}, got {spectrum_features.size(1)}")


        # bounded_residual: [B, output_dim], added either in logit space or probability space.
        bounded_residual = torch.tanh(self.net(spectrum_features))
        if self.residual_space == "logit":
            if spectrum_logits is None:
                if self.output_dim != self.input_dim:
                    raise ValueError("spectrum_logits are required when output_dim != input_dim.")
                clipped = spectrum_features.clamp(min=1.0e-6, max=1.0 - 1.0e-6)
                spectrum_logits = torch.logit(clipped)
            if spectrum_logits.size(1) != self.output_dim:
                raise ValueError(f"Expected spectrum_logits dim {self.output_dim}, got {spectrum_logits.size(1)}")

            refined_logits = spectrum_logits + self.residual_scale * bounded_residual
            refined_features = torch.sigmoid(refined_logits)
            base_features = torch.sigmoid(spectrum_logits)
        else:
            if self.output_dim != self.input_dim:
                raise ValueError("prob residual_space requires output_dim == input_dim.")
            refined_features = (spectrum_features + self.residual_scale * bounded_residual).clamp(
                min=0.0, max=1.0
            )
            refined_logits = None
            base_features = spectrum_features

        return {
            "features": refined_features,
            "residual": bounded_residual,
            "feature_shift": refined_features - base_features,
            "refined_logits": refined_logits,
        }


class SpectrumChannelCalibrator(nn.Module):


    def __init__(
        self,
        input_dim: int,
        lambda_scale: float = 0.0,
        lambda_bias: float = 0.02,
        use_scale: bool = False,
        use_bias: bool = True,
    ):
        super().__init__()
        if input_dim <= 0:
            raise ValueError("input_dim must be positive for SpectrumChannelCalibrator.")
        self.input_dim = int(input_dim)
        self.lambda_scale = float(lambda_scale)
        self.lambda_bias = float(lambda_bias)
        self.use_scale = bool(use_scale)
        self.use_bias = bool(use_bias)
        self.scale_raw = nn.Parameter(torch.zeros(self.input_dim))
        self.bias_raw = nn.Parameter(torch.zeros(self.input_dim))

    def forward(self, spectrum_logits: torch.Tensor):
        if spectrum_logits.ndim != 2:
            raise ValueError(
                f"Expected spectrum_logits to have shape [B, R], got {tuple(spectrum_logits.shape)}"
            )
        if spectrum_logits.size(1) != self.input_dim:
            raise ValueError(f"Expected spectrum dim {self.input_dim}, got {spectrum_logits.size(1)}")

        bounded_scale = torch.tanh(self.scale_raw).view(1, -1)
        bounded_bias = torch.tanh(self.bias_raw).view(1, -1)
        scale = 1.0
        if self.use_scale and self.lambda_scale != 0.0:
            scale = 1.0 + self.lambda_scale * bounded_scale
        bias = 0.0
        if self.use_bias and self.lambda_bias != 0.0:
            bias = self.lambda_bias * bounded_bias

        refined_logits = spectrum_logits * scale + bias
        refined_features = torch.sigmoid(refined_logits)
        base_features = torch.sigmoid(spectrum_logits)
        logit_shift = refined_logits - spectrum_logits

        reg_terms = []
        if self.use_scale:
            reg_terms.append(bounded_scale)
        if self.use_bias:
            reg_terms.append(bounded_bias)
        if reg_terms:
            regularizer = torch.cat([term.reshape(-1) for term in reg_terms], dim=0)
        else:
            regularizer = torch.zeros(1, dtype=spectrum_logits.dtype, device=spectrum_logits.device)

        return {
            "features": refined_features,
            "refined_logits": refined_logits,
            "feature_shift": refined_features - base_features,
            "logit_shift": logit_shift,
            "bounded_scale": bounded_scale,
            "bounded_bias": bounded_bias,
            "regularizer": regularizer,
        }


class M1VNew(nn.Module):


    def __init__(self, input_dim: int = 15, hidden_dim: int = 32, dropout: float = 0.1,
                 alpha_max: float = 0.5, b_max: float = 0.5):
        super().__init__()
        self.alpha_max = float(alpha_max)
        self.b_max = float(b_max)
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, rdkit_z: torch.Tensor, spectrum_summary: torch.Tensor):
        if rdkit_z.ndim != 2 or spectrum_summary.ndim != 2:
            raise ValueError("rdkit_z and spectrum_summary must both be rank-2 tensors.")

        # rdkit_z: [B, 7], spectrum_summary: [B, 8], concatenated input: [B, 15].
        x = torch.cat([rdkit_z, spectrum_summary], dim=-1)
        raw = self.net(x)
        o_alpha = raw[:, :1]
        o_b = raw[:, 1:2]

        # b is the learned scalar logit offset; alpha remains available for diagnostics.
        alpha = 1.0 + self.alpha_max * torch.tanh(o_alpha)
        b = self.b_max * torch.tanh(o_b)
        return {"alpha": alpha, "b": b, "raw": raw}


class ORSpectrumClassifier(nn.Module):


    def __init__(
        self,
        input_dim: int,
        num_labels: int,
        hidden_dim: int = 512,
        dropout: float = 0.2,
        use_3d_branch: bool = False,
        three_d_input_dim: int = 0,
        three_d_hidden_dim: int = 64,
        three_d_dropout: float = 0.1,
        three_d_norm_type: str = "layernorm",
        three_d_use_interaction_term: bool = True,
    ):
        super().__init__()
        self.use_3d_branch = bool(use_3d_branch)
        self.three_d_use_interaction_term = bool(three_d_use_interaction_term)

        if not self.use_3d_branch:
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, num_labels),
            )
            return

        if three_d_input_dim <= 0:
            raise ValueError("three_d_input_dim must be positive when use_3d_branch=True.")


        self.or_encoder = nn.Sequential(
            nn.Linear(input_dim, three_d_hidden_dim),
            build_norm_1d(three_d_norm_type, three_d_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.three_d_encoder = ThreeDFeatureEncoder(
            input_dim=three_d_input_dim,
            hidden_dim=three_d_hidden_dim,
            output_dim=three_d_hidden_dim,
            dropout=three_d_dropout,
            norm_type=three_d_norm_type,
        )

        fusion_dim = three_d_hidden_dim * 2
        if self.three_d_use_interaction_term:
            fusion_dim += three_d_hidden_dim
        self.fusion_head = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_labels),
        )

    def forward(self, spectrum_features: torch.Tensor, extra_feature: torch.Tensor = None) -> torch.Tensor:
        if self.use_3d_branch:
            if extra_feature is None:
                raise ValueError("extra_feature must be provided when use_3d_branch=True.")

            # spectrum_features: [B, R], extra_feature: [B, D3].
            h_or = self.or_encoder(spectrum_features)
            h_3d = self.three_d_encoder(extra_feature)
            fused_parts = [h_or, h_3d]
            if self.three_d_use_interaction_term:
                # Elementwise interaction lets OR-spectrum and 3D signals modulate each other.
                fused_parts.append(h_or * h_3d)
            fused = torch.cat(fused_parts, dim=-1)
            return self.fusion_head(fused)

        if extra_feature is not None:
            spectrum_features = torch.cat([spectrum_features, extra_feature], dim=-1)
        return self.net(spectrum_features)


def apply_m1_calibration(
    r0: torch.Tensor,
    alpha: torch.Tensor,
    b: torch.Tensor,
    mode: str = "affine",
):

    if mode not in {"none", "b_only", "affine"}:
        raise ValueError(f"Unsupported m1_mode: {mode}")
    mu = r0.mean(dim=1, keepdim=True)
    if mode == "none":
        r1 = r0
        alpha_used = torch.ones_like(mu)
        b_used = torch.zeros_like(mu)
    elif mode == "b_only":
        r1 = r0 + b
        alpha_used = torch.ones_like(alpha)
        b_used = b
    else:
        r1 = alpha * (r0 - mu) + mu + b
        alpha_used = alpha
        b_used = b
    p1 = torch.sigmoid(r1)
    return {"r1": r1, "p1": p1, "mu": mu, "alpha": alpha_used, "b": b_used}


def build_classifier_input(r1: torch.Tensor, p1: torch.Tensor, classifier_input: str) -> torch.Tensor:

    if classifier_input == "p1":
        return p1
    if classifier_input == "r1":
        return r1
    if classifier_input == "both":
        return torch.cat([r1, p1], dim=1)
    raise ValueError(f"Unsupported classifier_input: {classifier_input}")
