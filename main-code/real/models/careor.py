"""CaReOR model, Molecular Calibration, and multimodal M3 components."""
import torch
from torch import nn
from dgllife.model.model_zoo.mlp_predictor import MLPPredictor
from real.tools import runtime as c

ARMS = ("graph_only", "raw_or", "m1_or", "m1_or_3d", "full")


class MolecularCalibration(nn.Module):
    """Current seven-property M1: one shared, identity-initialized logit bias."""
    def __init__(self):
        super().__init__()
        self.property_encoder = nn.Sequential(
            nn.Linear(7, 96), nn.LayerNorm(96), nn.GELU(), nn.Dropout(0.05),
            nn.Linear(96, 96), nn.GELU(), nn.Dropout(0.05))
        self.bias_head = nn.Linear(96, 1)
        nn.init.zeros_(self.bias_head.weight)
        nn.init.zeros_(self.bias_head.bias)

    def forward(self, properties):
        c.require(properties.ndim == 2 and properties.shape[1] == 7, "M1 expects seven molecular properties")
        return 0.5 * torch.tanh(self.bias_head(self.property_encoder(properties)))


class LatentSpectrumM3(nn.Module):
    """Spectrum Reencoding: OR/geometry encoders, interaction and fusion."""
    def __init__(self):
        super().__init__()
        self.or_encoder = nn.Sequential(nn.Linear(1237, 64), nn.LayerNorm(64), nn.GELU(), nn.Dropout(0.1))
        self.three_d_encoder = nn.Sequential(
            nn.Linear(42, 64), nn.LayerNorm(64), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(64, 64), nn.LayerNorm(64), nn.GELU(), nn.Dropout(0.1))
        self.fusion = nn.Sequential(
            nn.Linear(704, 404), nn.LayerNorm(404), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(404, 404), nn.LayerNorm(404), nn.GELU(), nn.Dropout(0.1))

    def forward(self, graph_features, calibrated_logits, geometry):
        u = self.or_encoder(torch.sigmoid(calibrated_logits))
        v = self.three_d_encoder(geometry)
        return self.fusion(torch.cat((graph_features, u, v, u * v), dim=1))


class CaReORModel(nn.Module):
    def __init__(self, arm, seed):
        super().__init__()
        c.require(arm in ARMS, "Unknown main ablation arm")
        self.arm = arm
        self.use_or = arm != "graph_only"
        self.use_m1 = arm in ("m1_or", "m1_or_3d", "full")
        self.use_3d = arm in ("m1_or_3d", "full")
        self.use_m3 = arm == "full"
        base = c.make_model()
        self.gnn, self.readout, self.predict = base.gnn, base.readout, base.predict
        self.m1, self.m3 = None, None
        # Additional modules get reproducible streams without perturbing baseline init.
        with torch.random.fork_rng(devices=[]):
            if self.use_or and not self.use_m3:
                torch.random.default_generator.manual_seed(seed + 1000)
                self.predict = MLPPredictor(512 + 1237 + (42 if self.use_3d else 0), 128, 152, 0.05)
            if self.use_m1:
                torch.random.default_generator.manual_seed(seed + 2000)
                self.m1 = MolecularCalibration()
            if self.use_m3:
                torch.random.default_generator.manual_seed(seed + 3000)
                self.m3 = LatentSpectrumM3()
                self.predict = nn.Linear(404, 152)

    def forward(self, graph, node_features, receptor_logits=None, properties=None, geometry=None,
                return_diagnostics=False):
        g = self.readout(graph, self.gnn(graph, node_features))
        return self.forward_features(g, receptor_logits, properties, geometry, return_diagnostics)

    def forward_features(self, g, receptor_logits=None, properties=None, geometry=None,
                         return_diagnostics=False):
        if not self.use_or:
            logits = self.predict(g)
            return (logits, {}) if return_diagnostics else logits
        c.require(receptor_logits is not None and receptor_logits.shape[1] == 1237, "OR features missing")
        b = self.m1(properties) if self.use_m1 else receptor_logits.new_zeros((len(g), 1))
        r1 = receptor_logits + b
        if self.use_3d:
            c.require(geometry is not None and geometry.shape[1] == 42, "3D features missing")
        if self.use_m3:
            features = self.m3(g, r1, geometry)
        else:
            values = [g, torch.sigmoid(r1)]
            if self.use_3d:
                values.append(geometry)
            features = torch.cat(values, dim=1)
        logits = self.predict(features)
        return (logits, {"bias": b, "calibrated_logits": r1}) if return_diagnostics else logits

    def manifest(self):
        count = lambda module: 0 if module is None else sum(p.numel() for p in module.parameters())
        return {"arm": self.arm, "route_order": ["graph", "or", "m1", "3d", "m3"],
                "gcn_factory": "dgllife.model.GCNPredictor",
                "uses_or": self.use_or, "uses_m1": self.use_m1,
                "uses_3d": self.use_3d, "uses_m3": self.use_m3,
                "head_dimensions": [404, 152] if self.use_m3 else [512 + (1237 if self.use_or else 0) + (42 if self.use_3d else 0), 128, 152],
                "gnn_parameters": count(self.gnn), "graph_readout_parameters": count(self.readout),
                "classifier_parameters": count(self.predict), "m1_parameters": count(self.m1),
                "m3_parameters": count(self.m3), "total_parameters": count(self),
                "or_conversion": "sigmoid_continuous_no_rounding" if self.use_or else None,
                "m1_input": "seven_properties_current_train_scaled" if self.use_m1 else None,
                "m1_operation": "r1=r0+b; b=0.5*tanh(network(properties))" if self.use_m1 else None,
                "geometry_connection": "inside_full_m3" if self.use_m3 else "direct_concat_42" if self.use_3d else None}
