

import numpy as np
import torch
import torch.nn as nn
import pandas as pd
import sys
import time
from pathlib import Path

from dgllife.utils import EarlyStopping, Meter, SMILESToBigraph
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset

from utils import collate_molgraphs, load_model, predict_OR_feat

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from real.models.m1_vnew import (
    M1VNew,
    SpectrumResidualReEncoder,
    summarize_spectrum_logits,
)
from real.data.gslf_m1_dataset import (
    apply_three_d_standardizer,
    build_three_d_alignment,
    fit_three_d_standardizer,
    load_three_d_feature_dataframe,
)
from a2_logging import (
    append_epoch_outputs,
    build_split_row,
    get_log_paths,
    get_lr,
    print_epoch_summary,
    print_experiment_header,
    save_best_summary,
)


# Descriptor columns consumed by the M1 b-only calibration module.
M1_Z_COLUMNS = ['z_MW', 'z_MolLogP', 'z_TPSA', 'z_HBD', 'z_HBA', 'z_RotB', 'z_Ring']


def str2bool(value):
    if isinstance(value, bool):
        return value
    val = str(value).strip().lower()
    if val in {'1', 'true', 'yes', 'y', 't'}:
        return True
    if val in {'0', 'false', 'no', 'n', 'f'}:
        return False
    raise ValueError('Could not parse bool value: {}'.format(value))


def get_subset_indices(subset):
    if hasattr(subset, 'indices'):
        return np.asarray(subset.indices, dtype=np.int64)
    inferred = []
    for item_idx in range(len(subset)):
        datum = subset[item_idx]
        if isinstance(datum, tuple) and len(datum) > 0 and isinstance(datum[0], (int, np.integer)):
            inferred.append(int(datum[0]))
        else:
            raise ValueError('Could not infer subset indices for 3D scaler fitting.')
    return np.asarray(inferred, dtype=np.int64)


def filter_rare_label_tasks(dataset, min_pos_count=0, min_pos_ratio=0.0):
    min_pos_count = int(min_pos_count)
    min_pos_ratio = float(min_pos_ratio)
    if min_pos_count <= 0 and min_pos_ratio <= 0:
        return None
    if not hasattr(dataset, 'labels') or not hasattr(dataset, 'mask'):
        raise ValueError('Rare-label filtering requires dataset.labels and dataset.mask.')

    labels = dataset.labels
    masks = dataset.mask
    if not torch.is_tensor(labels):
        labels = torch.tensor(labels, dtype=torch.float32)
    if not torch.is_tensor(masks):
        masks = torch.tensor(masks, dtype=torch.float32)
    labels = labels.float()
    masks = masks.float()

    valid = masks > 0
    positives = ((labels > 0.5) & valid).sum(dim=0)
    valid_counts = valid.sum(dim=0).clamp(min=1)
    prevalence = positives.float() / valid_counts.float()
    keep = (positives >= min_pos_count) & (prevalence >= min_pos_ratio)
    if int(keep.sum().item()) == 0:
        raise ValueError(
            'Rare-label filtering would remove all tasks. '
            'min_pos_count={}, min_pos_ratio={}'.format(min_pos_count, min_pos_ratio)
        )

    keep_idx = torch.where(keep)[0]
    drop_idx = torch.where(~keep)[0]
    old_n_tasks = int(labels.shape[1])
    old_tasks = list(dataset.tasks) if hasattr(dataset, 'tasks') else [str(i) for i in range(old_n_tasks)]
    dataset.labels = labels[:, keep_idx]
    dataset.mask = masks[:, keep_idx]
    if hasattr(dataset, 'tasks'):
        dataset.tasks = [old_tasks[int(i)] for i in keep_idx.tolist()]
    dataset.n_tasks = int(keep_idx.numel())

    dropped_names = [old_tasks[int(i)] for i in drop_idx.tolist()]

    summary = {
        'old_n_tasks': old_n_tasks,
        'new_n_tasks': int(keep_idx.numel()),
        'dropped_n_tasks': int(drop_idx.numel()),
        'min_pos_count': min_pos_count,
        'min_pos_ratio': min_pos_ratio,
        'min_kept_positives': int(positives[keep_idx].min().item()),
        'max_dropped_positives': int(positives[drop_idx].max().item()) if drop_idx.numel() else 0,
        'keep_indices': keep_idx.tolist(),
        'drop_indices': drop_idx.tolist(),
        'dropped_names': dropped_names,
    }
    print(
        '[rare-label-filter] tasks {} -> {}, dropped {}, min_pos_count={}, min_pos_ratio={:.4f}'.format(
            summary['old_n_tasks'],
            summary['new_n_tasks'],
            summary['dropped_n_tasks'],
            min_pos_count,
            min_pos_ratio,
        )
    )
    print(
        '[rare-label-filter] min kept positives={}, max dropped positives={}'.format(
            summary['min_kept_positives'],
            summary['max_dropped_positives'],
        )
    )
    return summary


class IndexedSubsetForORFeatures(Dataset):
    def __init__(self, subset, max_node_len):
        self.subset = subset
        self.indices = get_subset_indices(subset)
        self.max_node_len = int(max_node_len)

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, item):
        original_idx = int(self.indices[item])
        datum = self.subset[item]
        if len(datum) == 7:
            return datum
        if len(datum) == 4:
            smiles, graph, labels, masks = datum
            node_mask = np.zeros(self.max_node_len, dtype=np.float32)
            node_mask[: min(int(graph.number_of_nodes()), self.max_node_len)] = 1.0

        if len(datum) == 5:
            smiles, graph, labels, masks, mol_id = datum
            node_mask = np.zeros(self.max_node_len, dtype=np.float32)
            node_mask[: min(int(graph.number_of_nodes()), self.max_node_len)] = 1.0
            return original_idx, smiles, graph, labels, masks, mol_id, node_mask
        raise ValueError('Unsupported dataset item format with {} fields.'.format(len(datum)))


def unpack_gslf_batch(batch_data):
    if len(batch_data) == 7:
        idxs, smiles, bg, labels, masks, ids, node_masks = batch_data
        idxs = [int(idx) for idx in idxs]
        return idxs, smiles, bg, labels, masks, ids, node_masks
    if len(batch_data) == 4:
        idxs, smiles, bg, labels, masks = batch_data[0], batch_data[1], batch_data[2], batch_data[3], None

        idxs = [int(idx) for idx in idxs]
        return idxs, smiles, bg, labels, masks, None, None
    if len(batch_data) == 5:
        idxs, smiles, bg, labels, masks = batch_data
        idxs = [int(idx) for idx in idxs]
        return idxs, smiles, bg, labels, masks, None, None
    raise ValueError('Unsupported batch format with {} fields.'.format(len(batch_data)))


def compute_masked_bce_loss(args, loss_criterion, logits, labels, masks):
    label_mask = (masks != 0).float()
    mode = args.get('loss_mask_mode', 'masked_average')
    if mode == 'legacy_mean':

        safe_labels = torch.nan_to_num(labels, nan=0.0, posinf=1.0, neginf=0.0)
        return (loss_criterion(logits, safe_labels) * label_mask).mean()
    if mode == 'masked_average':

        safe_labels = torch.where(label_mask > 0, labels, torch.zeros_like(labels))
        loss_matrix = loss_criterion(logits, safe_labels)
        return (loss_matrix * label_mask).sum() / label_mask.sum().clamp(min=1.0)
    raise ValueError('Unsupported loss_mask_mode: {}'.format(mode))


def extract_state_dict(ckpt_obj):
    if isinstance(ckpt_obj, dict):
        if 'model_state_dict' in ckpt_obj:
            return ckpt_obj['model_state_dict']
        if 'state_dict' in ckpt_obj:
            return ckpt_obj['state_dict']
    return ckpt_obj


def normalize_state_dict_prefix(state_dict):
    if any(str(key).startswith('module.') for key in state_dict.keys()):
        return {
            (str(key)[7:] if str(key).startswith('module.') else str(key)): value
            for key, value in state_dict.items()
        }
    return state_dict


def infer_predictor_io_dims(state_dict):
    pairs = []
    for key in state_dict.keys():
        key_str = str(key)
        if key_str.startswith('predict.predict.') and key_str.endswith('.weight'):
            layer_idx = int(key_str.split('.')[2])
            pairs.append((layer_idx, key_str))
    if not pairs:
        return None, None
    pairs.sort(key=lambda item: item[0])
    first_key = pairs[0][1]
    last_key = pairs[-1][1]
    in_dim = int(state_dict[first_key].shape[1]) if state_dict[first_key].ndim == 2 else None
    n_tasks = int(state_dict[last_key].shape[0]) if state_dict[last_key].ndim == 2 else None
    return in_dim, n_tasks


def infer_molor_runtime_hparams(state_dict, prot_dim):
    mol2prot_dim = False
    key_q1 = 'cross_attn.query_transform_tensor1.weight'
    if key_q1 in state_dict and state_dict[key_q1].ndim == 2:
        out_dim_q1 = int(state_dict[key_q1].shape[0])
        mol2prot_dim = out_dim_q1 == int(prot_dim)

    predictor_in_dim, n_tasks = infer_predictor_io_dims(state_dict)
    gnn_attended_feats = None
    if predictor_in_dim is not None and predictor_in_dim > prot_dim:

        gnn_attended_feats = int(predictor_in_dim - prot_dim)
    elif 'feat_norm.weight' in state_dict:
        feat_norm_dim = int(state_dict['feat_norm.weight'].shape[0])
        if feat_norm_dim > prot_dim:
            gnn_attended_feats = int(feat_norm_dim - prot_dim)
    if n_tasks is None:
        n_tasks = 1

    return mol2prot_dim, gnn_attended_feats, n_tasks


def load_rdkit_z_features(dataset, morvalue_csv):
    mor_df = pd.read_csv(morvalue_csv)
    missing_cols = [col for col in ['smiles'] + M1_Z_COLUMNS if col not in mor_df.columns]
    if missing_cols:
        raise ValueError('morvalue_csv is missing required columns: {}'.format(missing_cols))
    mor_df = mor_df.drop_duplicates(subset=['smiles'], keep='first')
    for col in M1_Z_COLUMNS:
        mor_df[col] = pd.to_numeric(mor_df[col], errors='coerce')
    non_finite_count = int(mor_df[M1_Z_COLUMNS].isna().any(axis=1).sum())
    if non_finite_count > 0:
        print('[warning] morvalue_csv has {} rows with non-finite RDKit z features; filling them with 0.0.'.format(non_finite_count))
        mor_df[M1_Z_COLUMNS] = mor_df[M1_Z_COLUMNS].fillna(0.0)
    feature_map = {
        smiles: row[M1_Z_COLUMNS].to_numpy(dtype=np.float32)
        for _, row in mor_df.iterrows()
        for smiles in [row['smiles']]
    }
    features = []
    missing_smiles = []
    for smiles in dataset.smiles:
        if smiles not in feature_map:
            missing_smiles.append(smiles)
            features.append(np.zeros(len(M1_Z_COLUMNS), dtype=np.float32))
        else:
            features.append(feature_map[smiles])
    if missing_smiles:
        print('[warning] Missing RDKit z features for {} smiles; filling them with 0.0. First few missing: {}'.format(
            len(missing_smiles), missing_smiles[:5]
        ))
    return torch.tensor(np.stack(features), dtype=torch.float32)


def load_three_d_features(dataset, train_set, args):
    if not args.get('use_3d_branch', False):
        return None, None

    three_d_df = load_three_d_feature_dataframe(args['three_d_feat_path'])
    feat_3d_raw, has_3d, feature_names, feature_stats = build_three_d_alignment(dataset.smiles, three_d_df)
    train_indices = get_subset_indices(train_set)
    # Fit the 3D feature scaler on the training molecules only.
    scaler_state = fit_three_d_standardizer(feat_3d_raw, has_3d, train_indices, feature_names)

    class _Bundle:
        pass

    bundle = _Bundle()
    bundle.feat_3d_raw = feat_3d_raw
    bundle.has_3d = has_3d
    bundle.three_d_feature_names = feature_names
    bundle.three_d_feature_stats = feature_stats
    apply_three_d_standardizer(
        bundle,
        scaler_state=scaler_state,
        add_success_flag=args['three_d_add_success_flag'],
        fallback_zero=args['three_d_fallback_zero'],
    )
    return bundle.feat_3d, {
        'scaler_state': scaler_state,
        'feature_stats': feature_stats,
        'model_feature_names': bundle.three_d_model_feature_names,
    }


def compute_m1_features(
    args,
    full_OR_logits,
    idxs,
    rdkit_z,
    m1_model,
):
    # r0: [B, R] cached OR logits; rdkit_z_batch: [B, 7].
    r0 = torch.nan_to_num(full_OR_logits[idxs, :], nan=0.0, posinf=30.0, neginf=-30.0)
    rdkit_z_batch = rdkit_z[idxs, :]
    summary = summarize_spectrum_logits(r0)
    m1_out = m1_model(rdkit_z_batch, summary)
    alpha = m1_out['alpha']
    b = m1_out['b']
    # Full CaReOR uses the b-only calibration: r1 = r0 + b.
    r1 = r0 + b
    alpha_used = torch.ones_like(alpha)
    return torch.sigmoid(r1), alpha_used, b, r0, r1, {}


def apply_m3_residual_reencoder(args, p1, r1, m3_reencoder, aux, m3_context=None):
    if not args.get('use_m3_residual', False):
        return p1, aux
    if m3_reencoder is None:
        raise ValueError('use_m3_residual=True requires an initialized M3 re-encoder.')
    m3_input = p1
    if args.get('m3_condition_on_3d', False):
        if m3_context is None:
            raise ValueError('m3_condition_on_3d=True requires 3D context features.')
        m3_input = torch.cat([p1, m3_context.to(p1.device)], dim=1)
    # M3 learns a bounded residual in OR-spectrum space before percept prediction.
    m3_out = m3_reencoder(
        m3_input,
        spectrum_logits=r1 if args.get('m3_residual_space', 'logit') == 'logit' else None,
    )
    aux = dict(aux) if aux is not None else {}
    aux['m3_residual_for_loss'] = m3_out['residual']
    aux['m3_residual_tensor'] = m3_out['residual'].detach()
    aux['m3_shift_tensor'] = m3_out['feature_shift'].detach()
    aux['m3_input_mean'] = p1.detach().mean()
    aux['m3_output_mean'] = m3_out['features'].detach().mean()
    aux['m3_input_dim'] = float(m3_input.shape[1])
    return m3_out['features'], aux


def get_or_features(
    args,
    full_OR_logits,
    idxs,
    rdkit_z=None,
    m1_model=None,
    three_d_features=None,
    m3_reencoder=None,
):
    p1, alpha, b, r0, r1, geom_aux = compute_m1_features(
        args,
        full_OR_logits,
        idxs,
        rdkit_z,
        m1_model,
    )
    m3_context = None
    if args.get('m3_condition_on_3d', False):
        if three_d_features is None:
            raise ValueError('m3_condition_on_3d=True requires aggregated three_d_features.')
        m3_context = torch.nan_to_num(three_d_features[idxs, :].to(p1.device), nan=0.0, posinf=0.0, neginf=0.0)
    p1, geom_aux = apply_m3_residual_reencoder(args, p1, r1, m3_reencoder, geom_aux, m3_context=m3_context)
    if args.get('use_3d_branch', False) and three_d_features is not None:
        # Concatenate calibrated OR features [B, R] with aggregated 3D features [B, D3].
        p1 = torch.cat([p1, three_d_features[idxs, :].to(p1.device)], dim=1)
    stats = {
        'alpha_mean': alpha.mean().item(),
        'alpha_std': alpha.std(unbiased=False).item(),
        'alpha_min': alpha.min().item(),
        'alpha_max': alpha.max().item(),
        'b_mean': b.mean().item(),
        'b_std': b.std(unbiased=False).item(),
        'b_min': b.min().item(),
        'b_max': b.max().item(),
        'd_soft_before': torch.sigmoid(r0).mean().item(),
        'd_soft_after': p1.mean().item(),
    }
    if geom_aux and 'm3_residual_tensor' in geom_aux:
        m3_residual = geom_aux['m3_residual_tensor']
        m3_shift = geom_aux['m3_shift_tensor']
        stats.update({
            'm3_residual_mean': m3_residual.mean().item(),
            'm3_residual_std': m3_residual.std(unbiased=False).item(),
            'm3_residual_min': m3_residual.min().item(),
            'm3_residual_max': m3_residual.max().item(),
            'm3_shift_mean': m3_shift.mean().item(),
            'm3_shift_std': m3_shift.std(unbiased=False).item(),
            'm3_shift_min': m3_shift.min().item(),
            'm3_shift_max': m3_shift.max().item(),
            'm3_input_mean': float(geom_aux['m3_input_mean'].item()),
            'm3_output_mean': float(geom_aux['m3_output_mean'].item()),
        })
    return p1, stats, alpha, b, geom_aux

def run_a_train_epoch(args, epoch, model, OR_logits, data_loader, loss_criterion, optimizer,
                      metric=None, rdkit_z_features=None, m1_model=None, three_d_features=None,
                      m3_reencoder=None):
    train_m3_residual_head_only = args.get('train_m3_residual_head_only', False)
    train_m3_residual_fusion_only = args.get('train_m3_residual_fusion_only', False)
    if train_m3_residual_head_only or train_m3_residual_fusion_only:

        model.eval()
        if train_m3_residual_head_only and hasattr(model, 'predict'):
            model.predict.train()
        if train_m3_residual_fusion_only:
            for module_name in ['readout', 'predict', 'or_encoder', 'three_d_encoder', 'fusion_head']:
                if hasattr(model, module_name):
                    getattr(model, module_name).train()
    else:
        model.train()
    if m1_model is not None:
        m1_model.eval() if (train_m3_residual_head_only or train_m3_residual_fusion_only) else m1_model.train()
    if m3_reencoder is not None:
        m3_reencoder.train()
    train_meter = Meter()
    stats_accumulator = []
    total_loss = 0.0
    total_batches = 0
    for batch_id, batch_data in enumerate(data_loader):
        idxs, smiles, bg, labels, masks, ids, node_masks = unpack_gslf_batch(batch_data)
        if len(smiles) == 1:

            continue

        labels = labels.to(args['device'])
        masks = torch.ones_like(labels, device=args['device']) if masks is None else masks.to(args['device'])
        or_features, m1_stats, alpha, b, geom_aux = get_or_features(
            args, OR_logits, idxs, rdkit_z_features, m1_model, three_d_features,
            m3_reencoder=m3_reencoder,
        )
        or_features = torch.nan_to_num(or_features, nan=0.0, posinf=1.0, neginf=0.0)
        logits = predict_OR_feat(args, model, bg, or_features)
        logits = torch.nan_to_num(logits, nan=0.0, posinf=30.0, neginf=-30.0)

        loss = compute_masked_bce_loss(args, loss_criterion, logits, labels, masks)
        if args.get('use_m3_residual', False) and args.get('lambda_m3_db', 0.0) != 0.0:
            if geom_aux and 'm3_residual_for_loss' in geom_aux:

                loss = loss + float(args['lambda_m3_db']) * (geom_aux['m3_residual_for_loss'] ** 2).mean()
        loss = loss + args['lambda_alpha'] * ((alpha - 1.0) ** 2).mean()
        loss = loss + args['lambda_b'] * (b ** 2).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item())
        total_batches += 1
        train_meter.update(logits, labels, masks)
        if m1_stats is not None:
            stats_accumulator.append(m1_stats)
        if batch_id % args['print_every'] == 0:
            print('epoch {:d}/{:d}, batch {:d}/{:d}, loss {:.4f}'.format(
                epoch + 1, args['num_epochs'], batch_id + 1, len(data_loader), loss.item()))
    train_score = np.mean(train_meter.compute_metric(args['metric'] if metric is None else metric))
    print('epoch {:d}/{:d}, training {} {:.4f}'.format(
        epoch + 1, args['num_epochs'], args['metric'], train_score))
    if stats_accumulator:
        print_m1_stats('train', stats_accumulator)
    return {
        'score': float(train_score),
        'loss': float(total_loss / max(total_batches, 1)),
        'stats': summarize_m1_stats(stats_accumulator),
    }

def run_an_eval_epoch(args, model, OR_logits, data_loader, metric=None, rdkit_z_features=None, m1_model=None,
                      three_d_features=None, m3_reencoder=None, loss_criterion=None):
    model.eval()
    if m1_model is not None:
        m1_model.eval()
    if m3_reencoder is not None:
        m3_reencoder.eval()
    eval_meter = Meter()
    stats_accumulator = []
    total_loss = 0.0
    total_batches = 0
    with torch.no_grad():
        for batch_id, batch_data in enumerate(data_loader):
            idxs, smiles, bg, labels, masks, ids, node_masks = unpack_gslf_batch(batch_data)
            labels = labels.to(args['device'])
            masks = torch.ones_like(labels, device=args['device']) if masks is None else masks.to(args['device'])
            or_features, m1_stats, _, _, _ = get_or_features(
                args, OR_logits, idxs, rdkit_z_features, m1_model, three_d_features,
                m3_reencoder=m3_reencoder,
            )
            or_features = torch.nan_to_num(or_features, nan=0.0, posinf=1.0, neginf=0.0)
            logits = predict_OR_feat(args, model, bg, or_features)
            logits = torch.nan_to_num(logits, nan=0.0, posinf=30.0, neginf=-30.0)
            if loss_criterion is not None:
                loss = compute_masked_bce_loss(args, loss_criterion, logits, labels, masks)
                total_loss += float(loss.item())
                total_batches += 1
            eval_meter.update(logits, labels, masks)
            if m1_stats is not None:
                stats_accumulator.append(m1_stats)
    return (
        np.mean(eval_meter.compute_metric(args['metric'] if metric is None else metric)),
        stats_accumulator,
        float(total_loss / max(total_batches, 1)) if loss_criterion is not None else float('nan'),
    )


def print_m1_stats(prefix, stats_accumulator):
    summary = summarize_m1_stats(stats_accumulator)
    print(
        '{} alpha mean/std/min/max {:.4f}/{:.4f}/{:.4f}/{:.4f}'.format(
            prefix,
            summary['alpha_mean'], summary['alpha_std'],
            summary['alpha_min'], summary['alpha_max'],
        )
    )
    print(
        '{} b mean/std/min/max {:.4f}/{:.4f}/{:.4f}/{:.4f}'.format(
            prefix,
            summary['b_mean'], summary['b_std'],
            summary['b_min'], summary['b_max'],
        )
    )
    print(
        '{} d_soft before/after {:.6f}/{:.6f}'.format(
            prefix, summary['d_soft_before'], summary['d_soft_after']
        )
    )
    if 'm3_residual_mean' in summary:
        print(
            '{} M3 residual mean/std/min/max {:.4f}/{:.4f}/{:.4f}/{:.4f}'.format(
                prefix,
                summary['m3_residual_mean'], summary['m3_residual_std'],
                summary['m3_residual_min'], summary['m3_residual_max'],
            )
        )
        print(
            '{} M3 p shift mean/std/min/max {:.6f}/{:.6f}/{:.6f}/{:.6f}'.format(
                prefix,
                summary['m3_shift_mean'], summary['m3_shift_std'],
                summary['m3_shift_min'], summary['m3_shift_max'],
            )
        )
    if 'attention_context_dim' in summary:
        print(
            '{} attention context_dim r0_summary mean/std {:.0f}/{:.4f}/{:.4f}'.format(
                prefix,
                summary['attention_context_dim'],
                summary.get('r0_summary_mean', 0.0),
                summary.get('r0_summary_std', 0.0),
            )
        )


def summarize_m1_stats(stats_accumulator):
    if not stats_accumulator:
        return None
    summary = {}
    for key in stats_accumulator[0].keys():
        summary[key] = float(np.mean([stats[key] for stats in stats_accumulator]))
    return summary


def load_state_dict_from_checkpoint(module, checkpoint_path, device, label):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    missing, unexpected = module.load_state_dict(state_dict, strict=False)
    print('Loaded {} checkpoint: {}'.format(label, checkpoint_path))
    if missing:
        print('[warning] {} missing keys: {}'.format(label, missing))
    if unexpected:
        print('[warning] {} unexpected keys: {}'.format(label, unexpected))


def set_module_trainable(module, trainable):
    if module is None:
        return
    for param in module.parameters():
        param.requires_grad = bool(trainable)


def set_predict_head_trainable(model, trainable):
    if model is None:
        return
    if not hasattr(model, 'predict'):
        raise ValueError('train_m3_residual_head_only=True expects model.predict to exist.')
    set_module_trainable(model.predict, trainable)


def set_m3_fusion_readout_trainable(model, trainable):
    if model is None:
        return
    opened = []
    for module_name in ['readout', 'predict', 'or_encoder', 'three_d_encoder', 'fusion_head']:
        if hasattr(model, module_name):
            set_module_trainable(getattr(model, module_name), trainable)
            opened.append(module_name)
    if not opened:
        raise ValueError('train_m3_residual_fusion_only=True found no readout/predict/fusion modules.')
    print('M3 fusion-only trainable percept modules: {}'.format(', '.join(opened)))


def apply_final_careor_entry(args):
    # Keep the public entrypoint aligned with the Full CaReOR pipeline.
    fixed_values = {
        'dataset': 'GS_LF_OR',
        'model': 'GCN_OR',
        'featurizer_type': 'canonical',
        'split': 'random',
        'OR_database': 'M2OR',
        'num_OR_logits': 1237,
        'm1_mode': 'b_only',
        'use_3d_branch': True,
        'three_d_mode': 'aggregated',
        'use_m3_residual': True,
        'm3_residual_space': 'logit',
        'm3_condition_on_3d': False,
    }
    for key, value in fixed_values.items():
        args[key] = value
    if not args.get('three_d_feat_path', ''):
        args['three_d_feat_path'] = 'runs/three_d/gslf_3d_features.csv'
    return args


def main(args, exp_config, dataset, train_set, val_set, test_set):
    percept_model_name = args['model']
    if percept_model_name != 'GCN_OR':
        raise ValueError(
            'classification_OR_feat_ESM expects an OR-feature percept model '
            'GCN_OR, got {}.'.format(percept_model_name)
        )
    if 'max_node_len' not in args:
        if hasattr(dataset, 'max_node_len'):
            args['max_node_len'] = dataset.max_node_len
        elif hasattr(dataset, 'graphs'):
            args['max_node_len'] = max(graph.number_of_nodes() for graph in dataset.graphs)
        else:
            args['max_node_len'] = 100
    if args.get('m3_condition_on_3d', False):
        if not args.get('use_m3_residual', False):
            raise ValueError('m3_condition_on_3d=True requires --use_m3_residual true.')
        if not args.get('use_3d_branch', False) or args.get('three_d_mode', 'aggregated') != 'aggregated':
            raise ValueError('m3_condition_on_3d=True currently expects aggregated --use_3d_branch true.')
        if args.get('m3_residual_space', 'logit') != 'logit':
            raise ValueError('m3_condition_on_3d=True expects --m3_residual_space logit.')
    if args.get('train_m3_residual_head_only', False) and not args.get('use_m3_residual', False):
        raise ValueError('train_m3_residual_head_only=True requires --use_m3_residual true.')
    if args.get('train_m3_residual_fusion_only', False) and not args.get('use_m3_residual', False):
        raise ValueError('train_m3_residual_fusion_only=True requires --use_m3_residual true.')
    if args.get('train_m3_residual_head_only', False) and args.get('train_m3_residual_fusion_only', False):
        raise ValueError('Use either train_m3_residual_head_only or train_m3_residual_fusion_only, not both.')

    three_d_features = None
    three_d_meta = None
    if args.get('use_3d_branch', False):
        three_d_features, three_d_meta = load_three_d_features(dataset, train_set, args)

    if args.get('use_3d_branch', False):
        if three_d_features is not None:
            args['three_d_input_dim'] = int(three_d_features.shape[1])
        else:
            raise ValueError('use_3d_branch=True but no 3D feature source was loaded.')
    else:
        args['three_d_input_dim'] = 0

    if args['featurizer_type'] != 'pre_train':
        print(exp_config)
        print(args['node_featurizer'].feat_size())
        exp_config['in_node_feats'] = args['node_featurizer'].feat_size()
        if args['edge_featurizer'] is not None:
            exp_config['in_edge_feats'] = args['edge_featurizer'].feat_size()
    exp_config.update({
        'n_tasks': args['n_tasks'],
        'model': args['model']
    })

    data_loader = DataLoader(dataset=dataset, batch_size=exp_config['batch_size'], shuffle = False,
                                collate_fn=collate_molgraphs, num_workers=args['num_workers'])

    train_data = IndexedSubsetForORFeatures(train_set, args['max_node_len'])
    val_data = IndexedSubsetForORFeatures(val_set, args['max_node_len'])
    test_data = IndexedSubsetForORFeatures(test_set, args['max_node_len'])

    train_loader = DataLoader(dataset=train_data, batch_size=exp_config['batch_size'], shuffle=True,
                              collate_fn=collate_molgraphs, num_workers=args['num_workers'])
    val_loader = DataLoader(dataset=val_data, batch_size=exp_config['batch_size'],
                            collate_fn=collate_molgraphs, num_workers=args['num_workers'])
    test_loader = DataLoader(dataset=test_data, batch_size=exp_config['batch_size'],
                             collate_fn=collate_molgraphs, num_workers=args['num_workers'])

    exp_config['add_feat_size'] = args['num_OR_logits']
    if args.get('use_3d_branch', False):
        exp_config['add_feat_size'] += args['three_d_input_dim']

    exp_config['mol2prot_dim'] = args['mol2prot_dim']

    exp_config['max_node_len'] = args['max_node_len']


    if args['num_OR_logits'] > 0:
        import pandas as pd
        mol_OR = pd.read_csv('data/datasets/M2OR_original_mol_OR_pairs.csv', sep = ';')
        top_seqs = mol_OR['mutated_Sequence'].value_counts()[0:args['num_OR_logits']].keys().tolist()
        max_seq_len = len(max(top_seqs, key=len))
        seq_masks = torch.zeros((len(top_seqs), max_seq_len))
        print(seq_masks.shape)
        for i in range(len(top_seqs)):
            seq_masks[i, :len(top_seqs[i])] = 1
            top_seqs[i] += "<pad>"*(max_seq_len - len(top_seqs[i]))
        exp_config['max_seq_len'] = max_seq_len
        from data.m2or import esm_embed
        seq_embeddings = esm_embed(top_seqs, per_residue=True, random_weights=False, esm_model_version = '650m')
        print(len(seq_embeddings))
        seq_emb_arr = np.dstack(seq_embeddings)
        seq_embeddings = torch.FloatTensor(np.rollaxis(seq_emb_arr, -1))
        print(seq_embeddings.shape)
    else:
        exp_config['max_seq_len'] = 0

    model = load_model(exp_config).to(args['device'])
    loss_criterion = nn.BCEWithLogitsLoss(reduction='none')
    optimizer = Adam(model.parameters(), lr=exp_config['lr'],
                     weight_decay=exp_config['weight_decay'])
    stopper = EarlyStopping(patience=exp_config['patience'],
                            filename=args['result_path'] + '/model.pth',
                            metric=args['metric'])


    full_OR_logits = None
    args['or_logits_source'] = 'auto'






    if args.get('or_logits_path', ''):
        # Load cached MolOR logits with shape [num_molecules, num_ORs].
        if not os.path.isfile(args['or_logits_path']):
            raise FileNotFoundError('or_logits_path does not exist: {}'.format(args['or_logits_path']))
        print('Loading OR logits from explicit path: {}'.format(args['or_logits_path']))
        full_OR_logits = torch.load(args['or_logits_path'], map_location='cpu')
        if isinstance(full_OR_logits, dict):
            for key in ['logits', 'full_OR_logits', 'or_logits']:
                if key in full_OR_logits:
                    full_OR_logits = full_OR_logits[key]
                    break
        if not torch.is_tensor(full_OR_logits):
            raise ValueError('Expected --or_logits_path to contain a tensor or dict with logits tensor.')
        if full_OR_logits.ndim != 2:
            raise ValueError('Expected OR logits tensor with shape [num_molecules, num_ORs], got {}'.format(
                tuple(full_OR_logits.shape)
            ))
        if full_OR_logits.shape[0] != len(dataset):
            raise ValueError('OR logits molecule count {} does not match dataset size {}.'.format(
                full_OR_logits.shape[0], len(dataset)
            ))
        if full_OR_logits.shape[1] < args['num_OR_logits']:
            raise ValueError('OR logits has {} OR columns, fewer than requested num_OR_logits {}.'.format(
                full_OR_logits.shape[1], args['num_OR_logits']
            ))
        if full_OR_logits.shape[1] > args['num_OR_logits']:
            full_OR_logits = full_OR_logits[:, :args['num_OR_logits']]
        full_OR_logits = full_OR_logits.float()
        args['or_logits_source'] = args['or_logits_path']
    else:
        logits_path = 'data/datasets/full_{}_ORs_logits.pt'.format(args['num_OR_logits'])
        print("Loading logits from model trained on unweighed loss")
        if os.path.isfile(logits_path):
            full_OR_logits = torch.load(logits_path)
            args['or_logits_source'] = logits_path
        else:
            print("No logits file found")


    if full_OR_logits is None:
        # Fallback path for regenerating OR logits when the cache is absent.
        args['model'] = "MolOR"
        print('Generating OR logits')

        OR_checkpoint = torch.load(args['prev_model_path'] + '/model.pth', map_location=args['device'])
        OR_state_dict = normalize_state_dict_prefix(extract_state_dict(OR_checkpoint))
        inferred_mol2prot, inferred_gnn_attended_feats, inferred_prev_tasks = infer_molor_runtime_hparams(
            OR_state_dict, args['prot_dim']
        )

        exp_config['gnn_attended_feats'] = inferred_gnn_attended_feats

        exp_config.update({'n_tasks': inferred_prev_tasks})

        exp_config.update({'model': 'MolOR'})
        exp_config.update({'mol2prot_dim': inferred_mol2prot})
        exp_config.update({'add_feat_size' : args['prot_dim']})

        exp_config['num_gnn_layers'] = 2
        exp_config['predictor_hidden_feats'] = 128
        exp_config['gnn_hidden_feats'] = 256
        OR_model = load_model(exp_config).to(args['device'])

        OR_model.load_state_dict(OR_state_dict, strict=True)
        OR_model.eval()

        full_OR_logits = torch.zeros(len(dataset), args['num_OR_logits'], device=args['device'])

        with torch.no_grad():
            for batch_id, batch_data in enumerate(data_loader):
                idxs, smiles, bg, labels, masks, ids, node_masks = unpack_gslf_batch(batch_data)


                seq_masks = seq_masks.to(args['device'])
                if node_masks is not None:
                    node_masks = node_masks.to(args['device'])

                if len(smiles) == 1:

                    continue

                if batch_id % 4 == 0:
                    print('batch_id: ' + str(batch_id))

                labels = labels.to(args['device'])
                masks = torch.ones_like(labels, device=args['device']) if masks is None else masks.to(args['device'])



                for i in range(seq_embeddings.shape[0]):

                    seq_embed = seq_embeddings[i]


                    seq_mask = seq_masks[i]

                    seq_embed = seq_embed.repeat(len(smiles), 1, 1)
                    seq_mask = seq_mask.repeat(len(smiles), 1)
                    full_OR_logits[idxs, i] = predict_OR_feat(args, OR_model, bg, seq_embed, seq_mask, node_masks).squeeze(dim=1)

            print('done GS_LF OR logit predictions')
            print('Saving OR logits for 90-10 model trained on unweighed loss')
            torch.save(full_OR_logits, 'data/datasets/full_{}_ORs_logits.pt'.format(args['num_OR_logits']))
            args['or_logits_source'] = 'generated:data/datasets/full_{}_ORs_logits.pt'.format(args['num_OR_logits'])

    print(full_OR_logits.shape)
    print('OR logits source: {}'.format(args.get('or_logits_source', 'auto')))

    full_OR_logits = full_OR_logits.to(args['device'])
    rdkit_z_features = None
    m1_model = None
    m3_reencoder = None
    best_m3_path = args['result_path'] + '/m3_reencoder_seed_{}.pth'.format(args['seed'])
    rdkit_z_features = load_rdkit_z_features(dataset, args['morvalue_csv']).to(args['device'])
    m1_model = M1VNew(alpha_max=args['alpha_max'], b_max=args['b_max']).to(args['device'])
    print('Using baseline + M1 mode: {}'.format(args['m1_mode']))

    if args.get('use_m3_residual', False):
        m3_input_dim = int(full_OR_logits.shape[1])
        if args.get('m3_condition_on_3d', False):
            if three_d_features is None:
                raise ValueError('m3_condition_on_3d=True requires loaded aggregated 3D features.')
            m3_input_dim += int(three_d_features.shape[1])
        m3_reencoder = SpectrumResidualReEncoder(
            input_dim=m3_input_dim,
            hidden_dim=args['m3_hidden_dim'],
            dropout=args['m3_dropout'],
            norm_type=args['m3_norm_type'],
            residual_scale=args['m3_lambda'],
            residual_space=args['m3_residual_space'],
            output_dim=int(full_OR_logits.shape[1]),
        ).to(args['device'])
        print(
            'Using M3 spectrum residual re-encoder: space={}, lambda={}, input_dim={}, output_dim={}'.format(
                args['m3_residual_space'], args['m3_lambda'], m3_input_dim, int(full_OR_logits.shape[1])
            )
        )

    if args.get('load_percept_checkpoint', ''):
        load_state_dict_from_checkpoint(
            model,
            args['load_percept_checkpoint'],
            args['device'],
            'percept model',
        )
    if m1_model is not None and args.get('load_m1_checkpoint', ''):
        load_state_dict_from_checkpoint(
            m1_model,
            args['load_m1_checkpoint'],
            args['device'],
            'M1 model',
        )
    if args.get('train_m3_residual_head_only', False):
        set_module_trainable(model, False)
        set_predict_head_trainable(model, True)
        set_module_trainable(m1_model, False)
        set_module_trainable(m3_reencoder, True)
        print('Training M3 residual re-encoder + percept predict head only; GNN/M1/3D modules are frozen.')
    if args.get('train_m3_residual_fusion_only', False):
        set_module_trainable(model, False)
        set_m3_fusion_readout_trainable(model, True)
        set_module_trainable(m1_model, False)
        set_module_trainable(m3_reencoder, True)
        print('Training M3 residual re-encoder + percept readout/fusion modules only; GNN/M1 are frozen.')

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if m1_model is not None:
        trainable_params += [p for p in m1_model.parameters() if p.requires_grad]
    if m3_reencoder is not None:
        trainable_params += [p for p in m3_reencoder.parameters() if p.requires_grad]
    if not trainable_params:
        raise ValueError('No trainable parameters. Check freeze/load/M3 settings.')
    if args.get('train_m3_residual_head_only', False):
        optimizer_lr = float(args.get('m3_residual_head_lr', 0.0))
    elif args.get('train_m3_residual_fusion_only', False):
        optimizer_lr = float(args.get('m3_residual_fusion_lr', 0.0))
    else:
        optimizer_lr = exp_config['lr']
    if optimizer_lr <= 0:
        optimizer_lr = exp_config['lr']
    optimizer = Adam(
        trainable_params,
        lr=optimizer_lr,
        weight_decay=exp_config['weight_decay'],
    )

    exp_config.update({'n_tasks': args['n_tasks']})
    exp_config.update({'model': percept_model_name})
    args['model'] = percept_model_name
    best_m1_path = args['result_path'] + '/m1_model_seed_{}.pth'.format(args['seed'])
    best_model_seed_path = args['result_path'] + '/model_seed_{}.pth'.format(args['seed'])
    log_paths = get_log_paths(args)
    if args.get('save_epoch_csv', True):
        for stale_path in [log_paths['text'], log_paths['csv'], log_paths['jsonl'], log_paths['batch_jsonl']]:
            if stale_path.is_file():
                stale_path.unlink()
    print_experiment_header(
        args,
        exp_config,
        args.get('dataset', 'unknown'),
        {'train': len(train_set), 'val': len(val_set), 'test': len(test_set)},
        log_paths,
    )
    run_start_time = time.perf_counter()
    best_epoch = -1
    best_payload = None

    for epoch in range(args['num_epochs']):

        epoch_start_time = time.perf_counter()
        train_info = run_a_train_epoch(
            args, epoch, model, full_OR_logits, train_loader, loss_criterion, optimizer,
            rdkit_z_features=rdkit_z_features, m1_model=m1_model, three_d_features=three_d_features,
            m3_reencoder=m3_reencoder,
        )




        val_score, val_stats, val_loss = run_an_eval_epoch(
            args, model, full_OR_logits, val_loader,
            rdkit_z_features=rdkit_z_features, m1_model=m1_model, three_d_features=three_d_features,
            m3_reencoder=m3_reencoder,
            loss_criterion=loss_criterion,
        )
        val_prc_score, _, _ = run_an_eval_epoch(
            args, model, full_OR_logits, val_loader, metric='pr_auc_score',
            rdkit_z_features=rdkit_z_features, m1_model=m1_model, three_d_features=three_d_features,
            m3_reencoder=m3_reencoder,
        )
        test_score_epoch, test_stats_epoch, test_loss_epoch = run_an_eval_epoch(
            args, model, full_OR_logits, test_loader,
            rdkit_z_features=rdkit_z_features, m1_model=m1_model, three_d_features=three_d_features,
            m3_reencoder=m3_reencoder,
            loss_criterion=loss_criterion,
        )
        test_prc_score_epoch, _, _ = run_an_eval_epoch(
            args, model, full_OR_logits, test_loader, metric='pr_auc_score',
            rdkit_z_features=rdkit_z_features, m1_model=m1_model, three_d_features=three_d_features,
            m3_reencoder=m3_reencoder,
        )
        previous_best = stopper.best_score
        early_stop = stopper.step(val_score, model)
        epoch_time = time.perf_counter() - epoch_start_time
        total_time = time.perf_counter() - run_start_time
        train_summary = train_info.get('stats')
        val_stats_summary = summarize_m1_stats(val_stats)
        test_stats_summary_epoch = summarize_m1_stats(test_stats_epoch)
        lr_now = get_lr(optimizer)
        epoch_rows = [
            build_split_row(epoch + 1, 'train', train_info.get('loss'), train_info.get('score'), float('nan'), lr_now, epoch_time, total_time, train_summary),
            build_split_row(epoch + 1, 'val', val_loss, val_score, val_prc_score, lr_now, epoch_time, total_time, val_stats_summary),
            build_split_row(epoch + 1, 'test', test_loss_epoch, test_score_epoch, test_prc_score_epoch, lr_now, epoch_time, total_time, test_stats_summary_epoch),
        ]
        append_epoch_outputs(
            log_paths,
            epoch_rows,
            save_csv=args.get('save_epoch_csv', True),
            save_jsonl=args.get('save_epoch_csv', True),
        )
        if stopper.best_score != previous_best:
            best_epoch = epoch + 1
            best_payload = {
                'exp_name': args.get('exp_name', 'a2_debug'),
                'seed': int(args['seed']),
                'best_epoch': int(best_epoch),
                'best_val_{}'.format(args['metric']): float(stopper.best_score),
                'val_roc_auc': float(val_score),
                'val_pr_auc': float(val_prc_score),
                'test_roc_auc': float(test_score_epoch),
                'test_pr_auc': float(test_prc_score_epoch),
                'val_stats': val_stats_summary,
                'test_stats': test_stats_summary_epoch,
            }
            if args.get('save_best_summary', True):
                save_best_summary(log_paths, best_payload)
        print_epoch_summary(
            epoch + 1,
            epoch_rows[0],
            epoch_rows[1],
            epoch_rows[2],
            best_epoch,
            stopper.best_score,
            log_paths,
            verbose=args.get('log_verbose_stats', True),
        )
        print('epoch {:d}/{:d}, validation {} {:.4f}, validation {} {:.4f}, best validation {} {:.4f}'.format(
            epoch + 1, args['num_epochs'], args['metric'],
            val_score, 'prc_auc_score', val_prc_score,
            args['metric'], stopper.best_score))
        if val_stats:
            print_m1_stats('val', val_stats)
        if stopper.best_score != previous_best:
            torch.save({'model_state_dict': model.state_dict()}, best_model_seed_path)
        if m1_model is not None and stopper.best_score != previous_best:
            torch.save({'model_state_dict': m1_model.state_dict()}, best_m1_path)
        if m3_reencoder is not None and stopper.best_score != previous_best:
            torch.save({'model_state_dict': m3_reencoder.state_dict()}, best_m3_path)

        if early_stop:
            break

    stopper.load_checkpoint(model)
    if m1_model is not None and os.path.isfile(best_m1_path):
        m1_checkpoint = torch.load(best_m1_path, map_location=args['device'])
        m1_model.load_state_dict(m1_checkpoint['model_state_dict'])
    if m3_reencoder is not None and os.path.isfile(best_m3_path):
        m3_checkpoint = torch.load(best_m3_path, map_location=args['device'])
        m3_reencoder.load_state_dict(m3_checkpoint['model_state_dict'])
    val_score, val_stats, val_loss = run_an_eval_epoch(
        args, model, full_OR_logits, val_loader,
        rdkit_z_features=rdkit_z_features, m1_model=m1_model, three_d_features=three_d_features,
        m3_reencoder=m3_reencoder,
        loss_criterion=loss_criterion,
    )
    val_prc_score, _, _ = run_an_eval_epoch(
        args, model, full_OR_logits, val_loader, metric='pr_auc_score',
        rdkit_z_features=rdkit_z_features, m1_model=m1_model, three_d_features=three_d_features,
        m3_reencoder=m3_reencoder,
    )
    test_score, test_stats, test_loss = run_an_eval_epoch(
        args, model, full_OR_logits, test_loader,
        rdkit_z_features=rdkit_z_features, m1_model=m1_model, three_d_features=three_d_features,
        m3_reencoder=m3_reencoder,
        loss_criterion=loss_criterion,
    )
    test_prc_score, _, _ = run_an_eval_epoch(
        args, model, full_OR_logits, test_loader, metric='pr_auc_score',
        rdkit_z_features=rdkit_z_features, m1_model=m1_model, three_d_features=three_d_features,
        m3_reencoder=m3_reencoder,
    )
    print('val {} {:.4f}'.format(args['metric'], val_score))
    print('test {} {:.4f}'.format(args['metric'], test_score))
    print('val prc_auc_score {:.4f}'.format(val_prc_score))
    print('test prc_auc_score {:.4f}'.format(test_prc_score))
    val_stats_summary = summarize_m1_stats(val_stats)
    test_stats_summary = summarize_m1_stats(test_stats)
    if val_stats_summary is not None:
        print_m1_stats('best-val', val_stats)
    if test_stats_summary is not None:
        print_m1_stats('test', test_stats)

    with open(args['result_path'] + '/' + str(args['seed']) + '_eval.txt', 'w') as f:
        f.write('Best val {}: {}\n'.format(args['metric'], stopper.best_score))
        f.write('Val {}: {}\n'.format(args['metric'], val_score))
        f.write('Test {}: {}\n'.format(args['metric'], test_score))
        f.write('Val prc_auc_score: {}\n'.format(val_prc_score))
        f.write('Test prc_auc_score: {}\n'.format(test_prc_score))
        f.write('or_logits_source: {}\n'.format(args.get('or_logits_source', 'auto')))
        f.write('loss_mask_mode: {}\n'.format(args.get('loss_mask_mode', 'masked_average')))
        if args['m1_mode'] != 'off':
            f.write('M1 mode: {}\n'.format(args['m1_mode']))
        if args.get('use_m3_residual', False):
            f.write('use_m3_residual: True\n')
            f.write('m3_lambda: {}\n'.format(args['m3_lambda']))
            f.write('m3_hidden_dim: {}\n'.format(args['m3_hidden_dim']))
            f.write('m3_dropout: {}\n'.format(args['m3_dropout']))
            f.write('m3_norm_type: {}\n'.format(args['m3_norm_type']))
            f.write('m3_residual_space: {}\n'.format(args['m3_residual_space']))
            f.write('m3_condition_on_3d: {}\n'.format(args.get('m3_condition_on_3d', False)))
            f.write('lambda_m3_db: {}\n'.format(args['lambda_m3_db']))
            f.write('train_m3_residual_head_only: {}\n'.format(args.get('train_m3_residual_head_only', False)))
            f.write('m3_residual_head_lr: {}\n'.format(args.get('m3_residual_head_lr', 0.0)))
            f.write('train_m3_residual_fusion_only: {}\n'.format(args.get('train_m3_residual_fusion_only', False)))
            f.write('m3_residual_fusion_lr: {}\n'.format(args.get('m3_residual_fusion_lr', 0.0)))
        if args.get('use_3d_branch', False):
            f.write('use_3d_branch: True\n')
            f.write('three_d_mode: {}\n'.format(args['three_d_mode']))
            f.write('three_d_feat_path: {}\n'.format(args['three_d_feat_path']))
            f.write('three_d_input_dim: {}\n'.format(args['three_d_input_dim']))
            if three_d_meta is not None:
                f.write('three_d_valid_ratio: {}\n'.format(three_d_meta['feature_stats']['valid_3d_ratio']))
        if args.get('rare_label_summary') is not None:
            rare_summary = args['rare_label_summary']
            f.write('rare_label_filter: True\n')
            f.write('rare_label_min_pos_count: {}\n'.format(rare_summary['min_pos_count']))
            f.write('rare_label_min_pos_ratio: {}\n'.format(rare_summary['min_pos_ratio']))
            f.write('rare_label_old_n_tasks: {}\n'.format(rare_summary['old_n_tasks']))
            f.write('rare_label_new_n_tasks: {}\n'.format(rare_summary['new_n_tasks']))
            f.write('rare_label_dropped_n_tasks: {}\n'.format(rare_summary['dropped_n_tasks']))
            f.write('rare_label_dropped_names: {}\n'.format(','.join(rare_summary['dropped_names'])))
        if val_stats_summary is not None:
            f.write('Val alpha mean: {}\n'.format(val_stats_summary['alpha_mean']))
            f.write('Val alpha std: {}\n'.format(val_stats_summary['alpha_std']))
            f.write('Val alpha min: {}\n'.format(val_stats_summary['alpha_min']))
            f.write('Val alpha max: {}\n'.format(val_stats_summary['alpha_max']))
            f.write('Val b mean: {}\n'.format(val_stats_summary['b_mean']))
            f.write('Val b std: {}\n'.format(val_stats_summary['b_std']))
            f.write('Val b min: {}\n'.format(val_stats_summary['b_min']))
            f.write('Val b max: {}\n'.format(val_stats_summary['b_max']))
            f.write('Val d_soft_before: {}\n'.format(val_stats_summary['d_soft_before']))
            f.write('Val d_soft_after: {}\n'.format(val_stats_summary['d_soft_after']))
            if 'm3_residual_mean' in val_stats_summary:
                f.write('Val m3_residual mean: {}\n'.format(val_stats_summary['m3_residual_mean']))
                f.write('Val m3_residual std: {}\n'.format(val_stats_summary['m3_residual_std']))
                f.write('Val m3_shift mean: {}\n'.format(val_stats_summary['m3_shift_mean']))
                f.write('Val m3_shift std: {}\n'.format(val_stats_summary['m3_shift_std']))
        if test_stats_summary is not None:
            f.write('Test alpha mean: {}\n'.format(test_stats_summary['alpha_mean']))
            f.write('Test alpha std: {}\n'.format(test_stats_summary['alpha_std']))
            f.write('Test alpha min: {}\n'.format(test_stats_summary['alpha_min']))
            f.write('Test alpha max: {}\n'.format(test_stats_summary['alpha_max']))
            f.write('Test b mean: {}\n'.format(test_stats_summary['b_mean']))
            f.write('Test b std: {}\n'.format(test_stats_summary['b_std']))
            f.write('Test b min: {}\n'.format(test_stats_summary['b_min']))
            f.write('Test b max: {}\n'.format(test_stats_summary['b_max']))
            f.write('Test d_soft_before: {}\n'.format(test_stats_summary['d_soft_before']))
            f.write('Test d_soft_after: {}\n'.format(test_stats_summary['d_soft_after']))
            if 'm3_residual_mean' in test_stats_summary:
                f.write('Test m3_residual mean: {}\n'.format(test_stats_summary['m3_residual_mean']))
                f.write('Test m3_residual std: {}\n'.format(test_stats_summary['m3_residual_std']))
                f.write('Test m3_shift mean: {}\n'.format(test_stats_summary['m3_shift_mean']))
                f.write('Test m3_shift std: {}\n'.format(test_stats_summary['m3_shift_std']))

if __name__ == '__main__':
    import os


    from argparse import ArgumentParser

    from utils import init_featurizer, mkdir_p, split_dataset, get_configure

    parser = ArgumentParser('Multi-label Binary Classification')


    parser.add_argument('-d', '--dataset', choices=['GS_LF_OR'], default='GS_LF_OR',
                        help='Dataset to use (only M2OR and GS_LF are supported)')


    parser.add_argument('-mo', '--model', choices=['GCN_OR'], default='GCN_OR',
                        help='Model to use')
    parser.add_argument('-f', '--featurizer-type', choices=['canonical'], default='canonical',
                        help='Featurization for atoms (and bonds). This is required for models '
                             'other than gin_supervised_**.')
    parser.add_argument('-mol2prot', '--mol2prot_dim', action='store_true', default = False,
                        help= 'Before doing cross-attention, either map node_dim (usually 256) to prot_dim (usually 1280), \
                        or vice versa')
    parser.add_argument('-n_ORs', '--num_OR_logits', type=int, default=1237)
    parser.add_argument('-prot', '--prot_dim', type=int, default=1280)
    parser.add_argument('-pmp', '--prev_model_path', type=str, default='M2OR_Uniprot_original_GCN',
                        help = 'For model to generate OR logits, specify path to trained model to correctly load model.')

    parser.add_argument('-s', '--split', choices=['random'], default='random',
                        help='Dataset splitting method (default: scaffold)')
    parser.add_argument('-sr', '--split-ratio', default='0.8,0.1,0.1', type=str,
                        help='Proportion of the dataset to use for training, validation and test, '
                             '(default: 0.8,0.1,0.1)')
    parser.add_argument('-me', '--metric', choices=['roc_auc_score', 'pr_auc_score'],
                        default='roc_auc_score',
                        help='Metric for evaluation (default: roc_auc_score)')
    parser.add_argument('-n', '--num-epochs', type=int, default=1000,
                        help='Maximum number of epochs for training. '
                             'We set a large number by default as early stopping '
                             'will be performed. (default: 1000)')
    parser.add_argument('-OR_db', '--OR_database', choices=['M2OR'], default='M2OR',
                        help='Database to use for OR activations.')
    parser.add_argument('--or_logits_path', type=str, default='',
                        help='Optional path to a precomputed OR logits tensor [num_molecules, num_OR_logits].')
    parser.add_argument('-nw', '--num-workers', type=int, default=0,
                        help='Number of processes for data loading (default: 0)')
    parser.add_argument('-pe', '--print-every', type=int, default=20,
                        help='Print the training progress every X mini-batches')
    parser.add_argument('--loss_mask_mode', choices=['masked_average', 'legacy_mean'], default='masked_average',
                        help='Loss masking mode. masked_average keeps the current safe masked mean; legacy_mean reproduces the older mask*loss matrix mean.')
    parser.add_argument('--log_verbose_stats', type=str2bool, default=True)
    parser.add_argument('--save_epoch_csv', type=str2bool, default=True)
    parser.add_argument('--save_best_summary', type=str2bool, default=True)
    parser.add_argument('--exp_name', type=str, default='a2_debug')
    parser.add_argument('-gnn_attend', '--gnn_attended_feats', type=int, default=None)
    parser.add_argument('--m1_mode', choices=['b_only'], default='b_only')
    parser.add_argument('--morvalue_csv', type=str, default='../morvalue.csv')
    parser.add_argument('--alpha_max', type=float, default=0.5)
    parser.add_argument('--b_max', type=float, default=0.5)
    parser.add_argument('--lambda_alpha', type=float, default=1e-3)
    parser.add_argument('--lambda_b', type=float, default=1e-3)
    parser.add_argument('--use_3d_branch', type=str2bool, default=True)
    parser.add_argument('--three_d_mode', choices=['aggregated'], default='aggregated')
    parser.add_argument('--three_d_feat_path', type=str, default='runs/three_d/gslf_3d_features.csv')
    parser.add_argument('--three_d_fallback_zero', type=str2bool, default=True)
    parser.add_argument('--three_d_add_success_flag', type=str2bool, default=True)
    parser.add_argument('--use_m3_residual', type=str2bool, default=True,
                        help='Enable M3 spectrum-level residual re-encoding after M1/3D calibration.')
    parser.add_argument('--m3_lambda', type=float, default=0.1,
                        help='Scale for the bounded M3 residual.')
    parser.add_argument('--lambda_m3_db', type=float, default=0.0,
                        help='L2 regularization weight for M3 bounded residuals.')
    parser.add_argument('--m3_hidden_dim', type=int, default=256)
    parser.add_argument('--m3_dropout', type=float, default=0.1)
    parser.add_argument('--m3_norm_type', choices=['layernorm', 'batchnorm', 'none'], default='layernorm')
    parser.add_argument('--m3_residual_space', choices=['logit', 'prob'], default='logit')
    parser.add_argument('--m3_condition_on_3d', type=str2bool, default=False,
                        help='Condition M3 residual on concat([p1, aggregated 3D features]) while outputting OR residuals.')
    parser.add_argument('--train_m3_residual_head_only', type=str2bool, default=False,
                        help='Freeze GNN/M1 and train only M3 residual re-encoder plus model.predict head.')
    parser.add_argument('--m3_residual_head_lr', type=float, default=0.001,
                        help='Optimizer lr used when --train_m3_residual_head_only true.')
    parser.add_argument('--train_m3_residual_fusion_only', type=str2bool, default=False,
                        help='Freeze GNN/M1 and train M3 plus readout/predict/fusion modules when present.')
    parser.add_argument('--m3_residual_fusion_lr', type=float, default=0.0001,
                        help='Optimizer lr used when --train_m3_residual_fusion_only true.')
    parser.add_argument('--load_percept_checkpoint', type=str, default='',
                        help='Optional GS-LF percept classifier checkpoint to initialize from.')
    parser.add_argument('--load_m1_checkpoint', type=str, default='',
                        help='Optional M1VNew checkpoint to initialize from.')
    parser.add_argument('--min_label_pos_count', type=int, default=0,
                        help='Drop odor-label tasks with fewer positive samples than this count before splitting. Default 0 disables.')
    parser.add_argument('--min_label_pos_ratio', type=float, default=0.0,
                        help='Drop odor-label tasks with positive prevalence below this ratio before splitting. Default 0 disables.')
    parser.add_argument('--seeds', type=str, default='42')
    parser.add_argument('-rp', '--result-path', type=str, default='classification_results',
                        help='Path to save training results (default: classification_results)')
    args = parser.parse_args().__dict__
    args = apply_final_careor_entry(args)

    if torch.cuda.is_available():
        args['device'] = torch.device('cuda:0')
    else:
        args['device'] = torch.device('cpu')

    seeds = [int(seed.strip()) for seed in str(args.get('seeds', '42')).split(',') if seed.strip()]

    for seed in seeds:
        args['seed'] = seed
        print('SEED NO: ' + str(seed))
        torch.manual_seed(seed)

        args = init_featurizer(args)
        mkdir_p(args['result_path'])
        smiles_to_g = SMILESToBigraph(add_self_loop=True, node_featurizer=args['node_featurizer'],
                                    edge_featurizer=args['edge_featurizer'])

        from data.m2or import GS_LF_OR
        dataset = GS_LF_OR(smiles_to_graph=smiles_to_g,
                    n_jobs=1 if args['num_workers'] == 0 else args['num_workers'], load=True)
        args['max_node_len'] = dataset.max_node_len

        if 'max_node_len' not in args and hasattr(dataset, 'max_node_len'):
            args['max_node_len'] = dataset.max_node_len

        rare_label_summary = filter_rare_label_tasks(
            dataset,
            min_pos_count=args.get('min_label_pos_count', 0),
            min_pos_ratio=args.get('min_label_pos_ratio', 0.0),
        )
        args['rare_label_summary'] = rare_label_summary
        args['n_tasks'] = dataset.n_tasks
        train_set, val_set, test_set = split_dataset(args, dataset)
        exp_config = get_configure(args['model'], args['featurizer_type'], args['dataset'])



        main(args, exp_config, dataset, train_set, val_set, test_set)


