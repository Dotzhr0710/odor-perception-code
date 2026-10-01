"""Joint component training and staged complete-model optimization."""
import os
from pathlib import Path
import numpy as np
import torch
from dgllife.utils import Meter
from real.tools import runtime as c
from real.data.gslf_dataset import make_loaders, historical_properties
from real.models.careor import CaReORModel, MolecularCalibration, LatentSpectrumM3


def experiment():
    return c.read(c.ROOT / "data/configures/training.json")


def state_copy(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def save_torch(path, value):
    tmp = Path(path).with_name(Path(path).name + ".tmp-" + str(os.getpid()))
    with open(tmp, "wb") as handle:
        torch.save(value, handle)
    os.replace(str(tmp), str(path))


def state_hash(model, prefixes=None):
    import hashlib
    h = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        if prefixes is None or name.startswith(prefixes):
            array = value.detach().cpu().contiguous().numpy()
            h.update(name.encode()); h.update(str(array.dtype).encode()); h.update(str(array.shape).encode()); h.update(array.tobytes())
    return h.hexdigest()


def forward_batch(model, batch, device):
    smiles, graph, labels, masks, extras = batch
    graph = graph.to(device)
    logits = model(graph, graph.ndata.pop("h"), **{k: v.to(device) for k, v in extras.items()})
    return logits, labels.to(device), masks.to(device)


def evaluate_loader(model, loader, device, metric=None):
    model.eval()
    logits, labels, masks = [], [], []
    with torch.no_grad():
        for batch in loader:
            pred, y, mask = forward_batch(model, batch, device)
            logits.append(pred.cpu()); labels.append(y.cpu()); masks.append(mask.cpu())
    arrays = {name: torch.cat(items).numpy() for name, items in (("logits", logits), ("labels", labels), ("masks", masks))}
    c.require(np.isfinite(arrays["logits"]).all(), "Non-finite prediction")
    if metric:
        meter = Meter()
        meter.update(*(torch.from_numpy(arrays[k]) for k in ("logits", "labels", "masks")))
        return float(np.mean(meter.compute_metric(metric)))
    return c.metrics(**arrays), arrays


def train_joint(model, train_loader, val_loader, device, folder, smoke=False):
    from dgllife.utils import EarlyStopping
    cfg = experiment()["joint"]
    epochs = experiment()["smoke_epochs_per_phase"] if smoke else cfg["max_epochs"]
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
    loss_function = torch.nn.BCEWithLogitsLoss(reduction="none")
    stopper = EarlyStopping(patience=cfg["patience"], filename=os.path.relpath(str(folder / "model.pth")), metric="roc_auc_score")
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        train_meter = Meter()
        for batch in train_loader:
            if len(batch[0]) == 1:
                continue
            logits, labels, masks = forward_batch(model, batch, device)
            loss = (loss_function(logits, labels) * (masks != 0).float()).mean()
            c.require(torch.isfinite(loss).item(), "Non-finite training loss")
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_meter.update(logits, labels, masks)
        # Preserve baseline's training Meter and two separate val loader iterations.
        train_score = float(np.mean(train_meter.compute_metric("roc_auc_score")))
        roc = evaluate_loader(model, val_loader, device, "roc_auc_score")
        pr = evaluate_loader(model, val_loader, device, "pr_auc_score")
        c.require(np.isfinite(roc) and np.isfinite(pr), "Non-finite validation metric")
        stop = stopper.step(roc, model)
        history.append({"phase": "joint", "epoch": epoch, "val_roc_auc": roc, "val_pr_auc": pr, "train_roc_auc": train_score})
        c.write(folder / "history.json", history)
        print("joint epoch=%d valROC=%.9f valPR=%.9f" % (epoch, roc, pr), flush=True)
        if stop:
            break
    best = max(history, key=lambda row: row["val_roc_auc"])
    checkpoint = torch.load(folder / "model.pth", map_location=device)
    c.require(checkpoint["timestep"] == best["epoch"], "Joint selected epoch mismatch")
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    return best, history, [{"phase": "joint", "trainable": [n for n, p in model.named_parameters() if p.requires_grad]}]


def make_staged_model(seed, device):
    c.seed_all(seed)
    model = CaReORModel("full", seed)
    path = c.INPUT_PATHS["backbone"]
    c.require(c.sha(path) == experiment()["canonical_backbone_sha256"], "Canonical backbone checksum mismatch")
    state = torch.load(path, map_location="cpu")["model_state_dict"]
    for name in ("gnn", "readout"):
        prefix = name + "."
        getattr(model, name).load_state_dict({k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}, strict=True)
    # Preserve active-module initialization order; only graph weights are imported.
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        old_m1 = MolecularCalibration()
        old_m3 = LatentSpectrumM3()
        old_classifier = torch.nn.Linear(404, 152)
    model.m1.load_state_dict(old_m1.state_dict(), strict=True)
    model.m3.load_state_dict(old_m3.state_dict(), strict=True)
    model.predict.load_state_dict(old_classifier.state_dict(), strict=True)
    for module in (model.gnn, model.readout):
        module.eval()
        for p in module.parameters():
            p.requires_grad_(False)
    return model.to(device)


def cache_graph_features(model, dataset, subset, device):
    from torch.utils.data import DataLoader
    utils, _ = c.historical()
    loader = DataLoader(subset, batch_size=128, shuffle=False, collate_fn=utils.collate_molgraphs, num_workers=0)
    model.eval()
    output = []
    with torch.no_grad():
        for _, graph, _, _ in loader:
            graph = graph.to(device)
            output.append(model.readout(graph, model.gnn(graph, graph.ndata.pop("h"))).cpu())
    return torch.cat(output)


def staged_splits(model, dataset, sets, aux, device, names=("train", "val")):
    output = {}
    for name in names:
        subset = sets[("train", "val", "test").index(name)]
        ids = torch.tensor(np.asarray(subset.indices), dtype=torch.long)
        output[name] = {"graph": cache_graph_features(model, dataset, subset, device),
                        "receptor_logits": aux["receptor"][ids], "properties": aux["properties"][ids],
                        "geometry": aux["geometry"][ids], "labels": dataset.labels[ids], "masks": dataset.mask[ids],
                        "indices": ids}
    return output


def staged_predict(model, split, device):
    model.eval()
    logits = []
    with torch.no_grad():
        for start in range(0, len(split["indices"]), 128):
            batch = {k: v[start:start + 128].to(device) for k, v in split.items()}
            pred = model.forward_features(batch["graph"], batch["receptor_logits"], batch["properties"], batch["geometry"])
            logits.append(pred.cpu())
    arrays = {"logits": torch.cat(logits).numpy(), "labels": split["labels"].numpy(), "masks": split["masks"].numpy()}
    c.require(np.isfinite(arrays["logits"]).all(), "Non-finite staged prediction")
    return c.metrics(**arrays), arrays


def configure_staged(model, phase):
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    if phase in ("warmup", "adapt"):
        for module in (model.m3, model.predict):
            module.train()
            for p in module.parameters():
                p.requires_grad_(True)
    if phase in ("m1", "adapt"):
        model.m1.train()
        for p in model.m1.parameters():
            p.requires_grad_(True)


def train_staged(model, splits, seed, device, folder, smoke=False):
    from real.tools.losses import build_train_label_statistics, direct_task_loss
    c.require(set(splits) == {"train", "val"}, "Training accepts train/val only")
    cfg = experiment()["staged"]
    stats = build_train_label_statistics(splits["train"]["labels"], splits["train"]["masks"])
    metric, _ = staged_predict(model, splits["val"], device)
    best = {"phase": "epoch_zero", "epoch": 0, "val_roc_auc": metric["roc_auc"], "val_pr_auc": metric["pr_auc"]}
    best_state = state_copy(model)
    history, audit = [dict(best)], []
    frozen_before = state_hash(model, ("gnn.", "readout."))
    for phase_index, phase in enumerate(("warmup", "m1", "adapt")):
        model.load_state_dict(best_state, strict=True)
        configure_staged(model, phase)
        groups = []
        for module in (model.m3, model.predict):
            params = [p for p in module.parameters() if p.requires_grad]
            if params:
                groups.append({"params": params, "lr": cfg["readout_learning_rate"] if phase == "warmup" else cfg["adapt_learning_rate"]})
        if phase in ("m1", "adapt"):
            groups.append({"params": list(model.m1.parameters()), "lr": cfg["m1_learning_rate"]})
        optimizer = torch.optim.AdamW(groups, weight_decay=cfg["weight_decay"])
        trainable = [name for name, p in model.named_parameters() if p.requires_grad]
        maximum = experiment()["smoke_epochs_per_phase"] if smoke else cfg[phase + "_epochs"]
        patience = 0
        for epoch in range(1, maximum + 1):
            rng_seed = seed * 100000 + phase_index * 1000 + epoch
            torch.manual_seed(rng_seed)
            if device.type == "cuda":
                torch.cuda.manual_seed_all(rng_seed)
            order = torch.randperm(len(splits["train"]["indices"]), generator=torch.Generator().manual_seed(rng_seed))
            configure_staged(model, phase)
            train_loss, updates = 0., 0
            for start in range(0, len(order), 128):
                ids = order[start:start + 128]
                if len(ids) == 1:
                    continue
                b = {k: v[ids].to(device) for k, v in splits["train"].items()}
                optimizer.zero_grad(set_to_none=True)
                logits, extra = model.forward_features(b["graph"], b["receptor_logits"], b["properties"], b["geometry"], True)
                loss, _ = direct_task_loss(logits, b["labels"], b["masks"], stats, experiment()["staged_loss"])
                loss = loss + cfg["m1_bias_energy_weight"] * extra["bias"].square().mean()
                c.require(torch.isfinite(loss).item(), "Non-finite staged loss")
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], cfg["gradient_clip_norm"])
                c.require(torch.isfinite(norm).item(), "Non-finite staged gradients")
                optimizer.step()
                train_loss += float(loss.detach()); updates += 1
            metric, _ = staged_predict(model, splits["val"], device)
            entry = {"phase": phase, "epoch": epoch, "val_roc_auc": metric["roc_auc"], "val_pr_auc": metric["pr_auc"],
                     "train_loss": train_loss / max(updates, 1), "updates": updates}
            history.append(entry)
            if entry["val_roc_auc"] > best["val_roc_auc"]:
                best, best_state, patience = dict(entry), state_copy(model), 0
            else:
                patience += 1
            c.write(folder / "history.json", history)
            print("%s epoch=%d valROC=%.9f valPR=%.9f" % (phase, epoch, metric["roc_auc"], metric["pr_auc"]), flush=True)
            if not smoke and patience >= cfg["patience"]:
                break
        audit.append({"phase": phase, "epochs": epoch, "trainable": trainable, "selected_after_phase": best})
    model.load_state_dict(best_state, strict=True)
    c.require(state_hash(model, ("gnn.", "readout.")) == frozen_before, "Frozen GCN/readout changed")
    save_torch(folder / "model.pth", {"model_state_dict": state_copy(model), "selected": best})
    return best, history, audit
