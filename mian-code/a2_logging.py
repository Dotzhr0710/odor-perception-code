

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd
import torch


M1_DEBUG_COLUMNS = ['S', 'S_prime', 's_hat', 'b']
SATURATION_EPS = 1e-3
SATURATION_WARN_RATIO = 0.20
B_FINAL_ABS_WARN_THRESHOLD = 3.0


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = str(value).strip().lower()
    if value in {'1', 'true', 'yes', 'y', 't'}:
        return True
    if value in {'0', 'false', 'no', 'n', 'f'}:
        return False
    raise ValueError('Could not parse bool value: {}'.format(value))


def load_explicit_m1_debug_features(dataset, morvalue_csv, device=None):
    mor_df = pd.read_csv(morvalue_csv)
    required_cols = ['smiles'] + M1_DEBUG_COLUMNS
    missing_cols = [col for col in required_cols if col not in mor_df.columns]
    if missing_cols:
        raise ValueError('morvalue_csv is missing required A2 debug columns: {}'.format(missing_cols))

    mor_df = mor_df.drop_duplicates(subset=['smiles'], keep='first')
    for col in M1_DEBUG_COLUMNS:
        mor_df[col] = pd.to_numeric(mor_df[col], errors='coerce')
    non_finite_rows = int(mor_df[M1_DEBUG_COLUMNS].isna().any(axis=1).sum())
    if non_finite_rows > 0:
        print('[warning] morvalue_csv has {} rows with non-finite A2 debug features; filling them with 0.0.'.format(
            non_finite_rows
        ))
        mor_df[M1_DEBUG_COLUMNS] = mor_df[M1_DEBUG_COLUMNS].fillna(0.0)

    feature_map = {
        row['smiles']: row[M1_DEBUG_COLUMNS].to_numpy(dtype=np.float32)
        for _, row in mor_df.iterrows()
    }
    missing_smiles = []
    features = []
    for smiles in dataset.smiles:
        value = feature_map.get(smiles)
        if value is None:
            missing_smiles.append(smiles)
            value = np.zeros(len(M1_DEBUG_COLUMNS), dtype=np.float32)
        features.append(value)
    if missing_smiles:
        print('[warning] Missing A2 debug features for {} smiles; filling them with 0.0. First few missing: {}'.format(
            len(missing_smiles), missing_smiles[:5]
        ))

    tensor = torch.tensor(np.stack(features), dtype=torch.float32)
    if device is not None:
        tensor = tensor.to(device)
    return tensor


def should_log_step(log_interval_steps, global_step):
    return int(log_interval_steps) > 0 and int(global_step) % int(log_interval_steps) == 0


def _tensor_stats(tensor):
    tensor = torch.nan_to_num(tensor.detach().reshape(-1), nan=0.0, posinf=0.0, neginf=0.0)
    return {
        'mean': float(tensor.mean().item()),
        'std': float(tensor.std(unbiased=False).item()),
        'min': float(tensor.min().item()),
        'max': float(tensor.max().item()),
    }


def _tensor_stats_full(tensor):
    tensor = torch.nan_to_num(tensor.detach().reshape(-1), nan=0.0, posinf=0.0, neginf=0.0)
    if tensor.numel() == 0:
        return {
            'mean': float('nan'),
            'std': float('nan'),
            'min': float('nan'),
            'max': float('nan'),
            'p5': float('nan'),
            'p50': float('nan'),
            'p95': float('nan'),
        }
    q = torch.quantile(tensor.float(), torch.tensor([0.05, 0.50, 0.95], device=tensor.device))
    return {
        'mean': float(tensor.mean().item()),
        'std': float(tensor.std(unbiased=False).item()),
        'min': float(tensor.min().item()),
        'max': float(tensor.max().item()),
        'p5': float(q[0].item()),
        'p50': float(q[1].item()),
        'p95': float(q[2].item()),
    }


def _add_stats(row, prefix, tensor, full=True):
    if tensor is None:
        return
    stats = _tensor_stats_full(tensor) if full else _tensor_stats(tensor)
    for key, value in stats.items():
        row['{}_{}'.format(prefix, key)] = value


def summarize_stat_rows(rows):
    rows = list(rows)
    if not rows:
        return {}
    keys = sorted({key for row in rows for key in row.keys()})
    summary = {}
    for key in keys:
        values = [row[key] for row in rows if key in row and np.isfinite(row[key])]
        summary[key] = float(np.mean(values)) if values else float('nan')
    return summary


def _warn_if_non_finite(name, tensor, warnings):
    if tensor is None:
        return
    detached = tensor.detach()
    if torch.isnan(detached).any():
        warnings.append('{} contains NaN'.format(name))
    if torch.isinf(detached).any():
        warnings.append('{} contains Inf'.format(name))


def _warn_if_saturated_s_hat(s_hat_tensor, warnings):
    if s_hat_tensor is None:
        return
    detached = s_hat_tensor.detach().reshape(-1)
    if detached.numel() == 0:
        return
    near_low = (detached <= (0.05 + SATURATION_EPS)).float().mean().item()
    near_high = (detached >= (0.95 - SATURATION_EPS)).float().mean().item()
    if near_low > SATURATION_WARN_RATIO or near_high > SATURATION_WARN_RATIO:
        warnings.append(
            's_hat shows saturation (low_ratio={:.3f}, high_ratio={:.3f})'.format(
                near_low, near_high
            )
        )


def build_a2_batch_debug_payload(
    batch_context,
    b_final,
    b_exp,
    delta_b_tilde=None,
    kappa_tilde=None,
):
    payload = {}
    if batch_context is not None:
        payload['S'] = batch_context[:, 0:1]
        payload['S_prime'] = batch_context[:, 1:2]
        payload['s_hat'] = batch_context[:, 2:3]
        payload['b_exp_from_context'] = batch_context[:, 3:4]
    payload['b_exp'] = b_exp
    payload['b_final'] = b_final
    if b_exp is not None and b_final is not None:
        payload['b_shift'] = b_final - b_exp
    if delta_b_tilde is not None:
        payload['delta_b_tilde'] = delta_b_tilde
    if kappa_tilde is not None:
        payload['kappa_tilde'] = kappa_tilde
    return payload


def build_a2_stats_from_payload(payload, delta_b_base=0.6, kappa_base=1.2):
    row = {}
    for name in ['S', 'S_prime', 's_hat', 'b_exp', 'b_final', 'b_shift', 'delta_b_tilde', 'kappa_tilde']:
        _add_stats(row, name, payload.get(name), full=True)

    if payload.get('delta_b_tilde') is not None:
        denom = max(abs(float(delta_b_base)), 1e-8)
        _add_stats(row, 'delta_b_tilde_over_base', payload['delta_b_tilde'] / denom, full=True)
    if payload.get('kappa_tilde') is not None:
        denom = max(abs(float(kappa_base)), 1e-8)
        _add_stats(row, 'kappa_tilde_over_base', payload['kappa_tilde'] / denom, full=True)

    s_hat = payload.get('s_hat')
    if s_hat is not None:
        flat = s_hat.detach().reshape(-1)
        row['saturation_low_ratio'] = float((flat < 0.05).float().mean().item()) if flat.numel() else float('nan')
        row['saturation_high_ratio'] = float((flat > 0.95).float().mean().item()) if flat.numel() else float('nan')

    b_final = payload.get('b_final')
    if b_final is not None:
        flat = b_final.detach().reshape(-1)
        row['b_final_extreme_ratio'] = float((flat.abs() > B_FINAL_ABS_WARN_THRESHOLD).float().mean().item()) if flat.numel() else float('nan')
    return row


def print_a2_batch_debug_log(
    epoch,
    global_step,
    batch_id,
    num_batches,
    loss_value,
    optimizer,
    payload,
    text_path=None,
    jsonl_path=None,
):
    lr = float(optimizer.param_groups[0]['lr']) if optimizer.param_groups else 0.0
    pieces = [
        '[a2-debug] epoch={:d} step={} batch={:d}/{:d} loss={:.6f} lr={:.6g}'.format(
            int(epoch) + 1,
            int(global_step),
            int(batch_id) + 1,
            int(num_batches),
            float(loss_value),
            lr,
        )
    ]
    warnings = []

    for name in ['S', 'S_prime']:
        tensor = payload.get(name)
        if tensor is not None:
            stats = _tensor_stats(tensor)
            pieces.append('{} mean/std={:.4f}/{:.4f}'.format(name, stats['mean'], stats['std']))
            _warn_if_non_finite(name, tensor, warnings)

    s_hat = payload.get('s_hat')
    if s_hat is not None:
        stats = _tensor_stats(s_hat)
        pieces.append('s_hat mean/std/min/max={:.4f}/{:.4f}/{:.4f}/{:.4f}'.format(
            stats['mean'], stats['std'], stats['min'], stats['max']
        ))
        _warn_if_non_finite('s_hat', s_hat, warnings)
        _warn_if_saturated_s_hat(s_hat, warnings)

    for name in ['b_exp', 'b_final', 'b_shift', 'delta_b_tilde', 'kappa_tilde']:
        tensor = payload.get(name)
        if tensor is None:
            continue
        stats = _tensor_stats(tensor)
        pieces.append('{} mean/std/min/max={:.4f}/{:.4f}/{:.4f}/{:.4f}'.format(
            name, stats['mean'], stats['std'], stats['min'], stats['max']
        ))
        _warn_if_non_finite(name, tensor, warnings)

    if not np.isfinite(float(loss_value)):
        warnings.append('loss is non-finite')

    message = ' | '.join(pieces)
    print(message)
    if text_path is not None:
        with open(text_path, 'a', encoding='utf-8') as handle:
            handle.write(message + '\n')
    for warning in warnings:
        warning_message = '[a2-debug][warning] {}'.format(warning)
        print(warning_message)
        if text_path is not None:
            with open(text_path, 'a', encoding='utf-8') as handle:
                handle.write(warning_message + '\n')
    if jsonl_path is not None:
        row = {
            'epoch': int(epoch) + 1,
            'global_step': int(global_step),
            'batch_id': int(batch_id) + 1,
            'num_batches': int(num_batches),
            'loss': float(loss_value),
            'lr': lr,
            'warnings': warnings,
        }
        row.update(build_a2_stats_from_payload(payload))
        with open(jsonl_path, 'a', encoding='utf-8') as handle:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + '\n')


def get_lr(optimizer):
    return float(optimizer.param_groups[0]['lr']) if optimizer.param_groups else 0.0


def log_prefix(args, seed=None):
    exp_name = args.get('exp_name', 'a2_debug')
    mode = args.get('a2_mode', 'E0')
    seed_value = args.get('seed', seed)
    if seed_value is None:
        return '{}_{}'.format(exp_name, mode)
    return '{}_{}_seed{}'.format(exp_name, mode, seed_value)


def get_log_paths(args, seed=None):
    out_dir = Path(args.get('result_path', args.get('out_dir', '.')))
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = log_prefix(args, seed=seed)
    return {
        'text': out_dir / '{}.log'.format(prefix),
        'csv': out_dir / '{}_epoch_summary.csv'.format(prefix),
        'jsonl': out_dir / '{}_epoch_summary.jsonl'.format(prefix),
        'batch_jsonl': out_dir / '{}_batch_debug.jsonl'.format(prefix),
        'best': out_dir / '{}_best_summary.json'.format(prefix),
    }


def write_lines(lines, text_path=None):
    for line in lines:
        print(line)
    if text_path is not None:
        text_path = Path(text_path)
        text_path.parent.mkdir(parents=True, exist_ok=True)
        with open(text_path, 'a', encoding='utf-8') as handle:
            for line in lines:
                handle.write(str(line) + '\n')


def print_experiment_header(args, exp_config, dataset_name, split_sizes, paths):
    lines = [
        '[Experiment]',
        '  exp_name: {}'.format(args.get('exp_name', 'a2_debug')),
        '  a2_mode: {}'.format(args.get('a2_mode', 'E0')),
        '  seed: {}'.format(args.get('seed', 'multi')),
        '  dataset: {}'.format(dataset_name),
        '  split sizes train/val/test: {}/{}/{}'.format(
            split_sizes.get('train', 0), split_sizes.get('val', 0), split_sizes.get('test', 0)
        ),
        '  batch_size: {}'.format(exp_config.get('batch_size', 'NA')),
        '  lr: {}'.format(exp_config.get('lr', 'NA')),
        '  weight_decay: {}'.format(exp_config.get('weight_decay', 'NA')),
        '  epochs: {}'.format(args.get('num_epochs', args.get('epochs', 'NA'))),
        '  loss_mask_mode: {}'.format(args.get('loss_mask_mode', 'masked_average')),
        '  lambda_geom: {}'.format(args.get('lambda_geom', 'NA')),
        '  lambda_db: {}'.format(args.get('lambda_db', 'NA')),
        '  use_3d_branch: {}'.format(args.get('use_3d_branch', False)),
        '  use_a2_correction: {}'.format(args.get('use_a2_correction', False)),
        '  a2_target: {}'.format(args.get('a2_target', args.get('a2_mode', 'E0'))),
        '  delta_b_base: {}'.format(args.get('delta_b_base', 0.6)),
        '  kappa_base: {}'.format(args.get('kappa_base', 1.2)),
        '  gamma_base: {}'.format(args.get('gamma_base', 2.0)),
        '  three_d_feat_path: {}'.format(args.get('three_d_feat_path', '')),
        '  three_d_input_dim: {}'.format(args.get('three_d_input_dim', 'NA')),
        '  result_path: {}'.format(args.get('result_path', args.get('out_dir', ''))),
        '  log_file: {}'.format(paths['text']),
        '  epoch_csv: {}'.format(paths['csv']),
        '  checkpoint_dir: {}'.format(args.get('result_path', args.get('out_dir', ''))),
    ]
    write_lines(lines, paths['text'])


def build_split_row(epoch, split, loss, roc_auc, pr_auc, lr, epoch_time, total_time, stats_summary):
    row = {
        'epoch': int(epoch),
        'split': split,
        'loss': float(loss) if loss is not None else float('nan'),
        'roc_auc': float(roc_auc) if roc_auc is not None else float('nan'),
        'pr_auc': float(pr_auc) if pr_auc is not None else float('nan'),
        'lr': float(lr),
        'epoch_time': float(epoch_time),
        'total_time': float(total_time),
    }
    if stats_summary:
        row.update(stats_summary)
    return row


def append_epoch_outputs(paths, rows, save_csv=True, save_jsonl=True):
    if not rows:
        return
    if save_jsonl:
        with open(paths['jsonl'], 'a', encoding='utf-8') as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + '\n')
    if save_csv:
        existing = None
        if paths['csv'].is_file():
            existing = pd.read_csv(paths['csv'])
        cur = pd.DataFrame(rows)
        out = cur if existing is None else pd.concat([existing, cur], ignore_index=True, sort=False)
        out.to_csv(paths['csv'], index=False, encoding='utf-8')


def print_epoch_summary(epoch, train_row, val_row, test_row, best_epoch, best_score, paths, verbose=True):
    lines = [
        '[Epoch Summary]',
        '  epoch: {} lr: {:.6g} epoch_time: {:.2f}s total_time: {:.2f}s'.format(
            epoch, val_row.get('lr', float('nan')), val_row.get('epoch_time', 0.0), val_row.get('total_time', 0.0)
        ),
        '[Val/Test Metrics]',
        '  train loss/roc/pr: {:.6f}/{:.6f}/{:.6f}'.format(
            train_row.get('loss', float('nan')), train_row.get('roc_auc', float('nan')), train_row.get('pr_auc', float('nan'))
        ),
        '  val loss/roc/pr: {:.6f}/{:.6f}/{:.6f}'.format(
            val_row.get('loss', float('nan')), val_row.get('roc_auc', float('nan')), val_row.get('pr_auc', float('nan'))
        ),
        '  test loss/roc/pr: {:.6f}/{:.6f}/{:.6f}'.format(
            test_row.get('loss', float('nan')), test_row.get('roc_auc', float('nan')), test_row.get('pr_auc', float('nan'))
        ),
        '[Checkpoint]',
        '  best_epoch: {} best_val: {:.6f}'.format(best_epoch, best_score),
    ]
    if verbose:
        lines.extend([
            '[A2 Stats]',
            '  val S_prime mean/std: {:.4f}/{:.4f}'.format(
                val_row.get('S_prime_mean', float('nan')), val_row.get('S_prime_std', float('nan'))
            ),
            '  val s_hat mean/std: {:.4f}/{:.4f}'.format(
                val_row.get('s_hat_mean', float('nan')), val_row.get('s_hat_std', float('nan'))
            ),
            '  val b_exp/b_final/delta mean: {:.4f}/{:.4f}/{:.4f}'.format(
                val_row.get('b_exp_mean', float('nan')),
                val_row.get('b_final_mean', float('nan')),
                val_row.get('b_shift_mean', float('nan')),
            ),
            '  val delta_b_tilde mean/std: {:.4f}/{:.4f}'.format(
                val_row.get('delta_b_tilde_mean', float('nan')),
                val_row.get('delta_b_tilde_std', float('nan')),
            ),
            '[Saturation Stats]',
            '  val s_hat low/high: {:.4f}/{:.4f} b_final_extreme: {:.4f}'.format(
                val_row.get('saturation_low_ratio', float('nan')),
                val_row.get('saturation_high_ratio', float('nan')),
                val_row.get('b_final_extreme_ratio', float('nan')),
            ),
        ])
    write_lines(lines, paths['text'])


def save_best_summary(paths, payload):
    with open(paths['best'], 'w', encoding='utf-8') as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
