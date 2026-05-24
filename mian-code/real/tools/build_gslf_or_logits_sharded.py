


import argparse
import glob
import os
import sys
from pathlib import Path

import numpy as np
import torch
from dgllife.utils import SMILESToBigraph
from torch.utils.data import DataLoader, Dataset


ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from classification_OR_feat_ESM import (
    extract_state_dict,
    infer_molor_runtime_hparams,
    normalize_state_dict_prefix,
    str2bool,
    unpack_gslf_batch,
)
from utils import collate_molgraphs, get_configure, init_featurizer, load_model, mkdir_p, predict_OR_feat


class IndexedFullDataset(Dataset):


    def __init__(self, dataset, max_node_len):
        self.dataset = dataset
        self.max_node_len = int(max_node_len)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        datum = self.dataset[idx]
        if len(datum) == 7 and isinstance(datum[0], (int, np.integer)):
            return datum
        if len(datum) == 4:
            smiles, graph, labels, masks = datum
            node_mask = np.zeros(self.max_node_len, dtype=np.float32)
            node_mask[: min(int(graph.number_of_nodes()), self.max_node_len)] = 1.0
            return int(idx), smiles, graph, labels, masks, None, node_mask
        if len(datum) == 5:
            smiles, graph, labels, masks, mol_id = datum
            node_mask = np.zeros(self.max_node_len, dtype=np.float32)
            node_mask[: min(int(graph.number_of_nodes()), self.max_node_len)] = 1.0
            return int(idx), smiles, graph, labels, masks, mol_id, node_mask
        raise ValueError('Unsupported dataset item format with {} fields.'.format(len(datum)))


def build_top_m2or_sequences(num_or_logits):
    mol_or = __import__('pandas').read_csv('data/datasets/M2OR_original_mol_OR_pairs.csv', sep=';')
    top_seqs = mol_or['mutated_Sequence'].value_counts()[0:num_or_logits].keys().tolist()
    max_seq_len = len(max(top_seqs, key=len))
    seq_masks = torch.zeros((len(top_seqs), max_seq_len), dtype=torch.float32)
    padded = []
    for i, seq in enumerate(top_seqs):
        seq_masks[i, :len(seq)] = 1.0
        padded.append(seq + '<pad>' * (max_seq_len - len(seq)))
    return padded, seq_masks


def build_or_sequence_cache(args):
    if args.OR_database != 'M2OR':
        raise ValueError('This sharded helper currently supports OR_database=M2OR only.')
    output = Path(args.seq_cache_pt)
    mkdir_p(str(output.parent))
    top_seqs, seq_masks = build_top_m2or_sequences(args.num_OR_logits)
    from data.m2or import esm_embed

    print('Embedding {} M2OR sequences with ESM {}...'.format(len(top_seqs), args.esm_version))
    seq_embeddings = esm_embed(
        top_seqs,
        per_residue=True,
        random_weights=False,
        esm_model_version=args.esm_version,
    )
    seq_emb_arr = np.dstack(seq_embeddings)
    seq_embeddings = torch.FloatTensor(np.rollaxis(seq_emb_arr, -1))
    payload = {
        'seq_embeddings': seq_embeddings.cpu(),
        'seq_masks': seq_masks.cpu(),
        'num_OR_logits': int(args.num_OR_logits),
        'max_seq_len': int(seq_masks.shape[1]),
        'OR_database': args.OR_database,
        'esm_version': args.esm_version,
    }
    torch.save(payload, str(output))
    print('Saved sequence cache: {}'.format(output))
    print('seq_embeddings shape: {}'.format(tuple(seq_embeddings.shape)))
    print('seq_masks shape: {}'.format(tuple(seq_masks.shape)))


def load_sequence_cache(args):
    if not os.path.isfile(args.seq_cache_pt):
        raise FileNotFoundError('seq_cache_pt does not exist: {}'.format(args.seq_cache_pt))
    payload = torch.load(args.seq_cache_pt, map_location='cpu')
    seq_embeddings = payload['seq_embeddings'].float()
    seq_masks = payload['seq_masks'].float()
    if seq_embeddings.shape[0] < args.or_end:
        raise ValueError(
            'Sequence cache has {} ORs but shard requests end {}.'.format(
                seq_embeddings.shape[0], args.or_end
            )
        )
    return seq_embeddings, seq_masks, int(payload.get('max_seq_len', seq_masks.shape[1]))


def load_gslf_dataset(args):
    featurizer_args = {
        'model': args.model,
        'featurizer_type': args.featurizer_type,
    }
    featurizer_args = init_featurizer(featurizer_args)
    smiles_to_g = SMILESToBigraph(
        add_self_loop=True,
        node_featurizer=featurizer_args['node_featurizer'],
        edge_featurizer=featurizer_args['edge_featurizer'],
    )
    if args.dataset != 'GS_LF_OR':
        raise ValueError('This helper is intended for dataset=GS_LF_OR.')
    from data.m2or import GS_LF_OR

    dataset = GS_LF_OR(
        smiles_to_graph=smiles_to_g,
        n_jobs=1 if args.num_workers == 0 else args.num_workers,
        load=args.load_cache,
    )
    max_node_len = dataset.max_node_len
    wrapped = IndexedFullDataset(dataset, max_node_len=max_node_len)
    loader = DataLoader(
        dataset=wrapped,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_molgraphs,
        num_workers=args.num_workers,
    )
    return dataset, loader, featurizer_args, max_node_len


def load_or_model(args, featurizer_args, max_node_len, max_seq_len):
    exp_config = get_configure(args.model, args.featurizer_type, args.dataset)
    exp_config['in_node_feats'] = featurizer_args['node_featurizer'].feat_size()
    exp_config['n_tasks'] = 1
    exp_config['model'] = 'MolOR'
    exp_config['max_node_len'] = int(max_node_len)
    exp_config['max_seq_len'] = int(max_seq_len)

    ckpt = torch.load(os.path.join(args.prev_model_path, 'model.pth'), map_location=args.device)
    state_dict = normalize_state_dict_prefix(extract_state_dict(ckpt))
    inferred_mol2prot, inferred_gnn_attended_feats, inferred_prev_tasks = infer_molor_runtime_hparams(
        state_dict,
        args.prot_dim,
    )

    exp_config.update({
        'model': 'MolOR',
        'n_tasks': inferred_prev_tasks,
        'add_feat_size': args.prot_dim,
        'mol2prot_dim': inferred_mol2prot,
        'gnn_attended_feats': inferred_gnn_attended_feats,
        'num_gnn_layers': 2,
        'predictor_hidden_feats': 128,
        'gnn_hidden_feats': 256,
    })
    model = load_model(exp_config).to(args.device)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model


def run_shard(args):
    if args.or_start < 0 or args.or_end <= args.or_start or args.or_end > args.num_OR_logits:
        raise ValueError('Invalid shard range [{}, {}) for num_OR_logits={}'.format(
            args.or_start, args.or_end, args.num_OR_logits
        ))
    mkdir_p(str(Path(args.output_pt).parent))
    args.device = torch.device('cuda:0' if torch.cuda.is_available() and args.device != 'cpu' else 'cpu')
    seq_embeddings, seq_masks, max_seq_len = load_sequence_cache(args)
    dataset, loader, featurizer_args, max_node_len = load_gslf_dataset(args)
    or_model = load_or_model(args, featurizer_args, max_node_len, max_seq_len)

    shard_width = int(args.or_end - args.or_start)
    shard_logits = torch.zeros(len(dataset), shard_width, dtype=torch.float32, device=args.device)
    seq_embeddings = seq_embeddings.to(args.device)
    seq_masks = seq_masks.to(args.device)
    pred_args = {
        'device': args.device,
        'edge_featurizer': featurizer_args['edge_featurizer'],
        'featurizer_type': args.featurizer_type,
        'model': 'MolOR',
    }

    print('Building OR-logit shard [{}, {}) on {}'.format(args.or_start, args.or_end, args.device))
    print('dataset molecules: {}, batch_size: {}'.format(len(dataset), args.batch_size))
    with torch.no_grad():
        for batch_id, batch_data in enumerate(loader):
            idxs, smiles, bg, labels, masks, ids, node_masks = unpack_gslf_batch(batch_data)
            if len(smiles) == 1:
                continue
            idxs = torch.as_tensor(idxs, dtype=torch.long, device=args.device)
            if node_masks is not None:
                node_masks = node_masks.to(args.device)
            if batch_id % args.print_every == 0:
                print('batch_id: {} / {}'.format(batch_id, len(loader)))
            for col, or_idx in enumerate(range(args.or_start, args.or_end)):
                seq_embed = seq_embeddings[or_idx].repeat(len(smiles), 1, 1)
                seq_mask = seq_masks[or_idx].repeat(len(smiles), 1)
                logits = predict_OR_feat(
                    pred_args,
                    or_model,
                    bg,
                    seq_embed,
                    seq_mask,
                    node_masks,
                ).squeeze(dim=1)
                shard_logits[idxs, col] = logits

    payload = {
        'logits': shard_logits.cpu(),
        'or_start': int(args.or_start),
        'or_end': int(args.or_end),
        'num_OR_logits': int(args.num_OR_logits),
        'num_molecules': int(len(dataset)),
    }
    torch.save(payload, args.output_pt)
    print('Saved shard: {}'.format(args.output_pt))
    print('shape: {}'.format(tuple(shard_logits.shape)))


def run_merge(args):
    paths = sorted(glob.glob(args.shard_glob))
    if not paths:
        raise FileNotFoundError('No shard files matched: {}'.format(args.shard_glob))
    parts = []
    for path in paths:
        payload = torch.load(path, map_location='cpu')
        parts.append((int(payload['or_start']), int(payload['or_end']), payload['logits'].float(), path))
    parts.sort(key=lambda item: item[0])
    expected = 0
    tensors = []
    for start, end, tensor, path in parts:
        if start != expected:
            raise ValueError('Shard gap/overlap before {}: expected start {}, got {}'.format(path, expected, start))
        expected = end
        tensors.append(tensor)
        print('merge shard [{}, {}) {} shape={}'.format(start, end, path, tuple(tensor.shape)))
    if expected != args.num_OR_logits:
        raise ValueError('Merged OR columns {}, expected {}'.format(expected, args.num_OR_logits))
    full = torch.cat(tensors, dim=1)
    mkdir_p(str(Path(args.output_pt).parent))
    torch.save(full, args.output_pt)
    print('Saved merged OR logits: {}'.format(args.output_pt))
    print('shape: {}'.format(tuple(full.shape)))
    print('finite: {}'.format(bool(torch.isfinite(full).all().item())))


def parse_args():
    parser = argparse.ArgumentParser('Build GS-LF OR logits in shards')
    parser.add_argument('mode', choices=['seq_cache', 'shard', 'merge'])
    parser.add_argument('-d', '--dataset', default='GS_LF_OR')
    parser.add_argument('-mo', '--model', default='GCN_OR')
    parser.add_argument('-f', '--featurizer_type', default='canonical')
    parser.add_argument('-n_ORs', '--num_OR_logits', type=int, default=1237)
    parser.add_argument('-OR_db', '--OR_database', default='M2OR')
    parser.add_argument('-pmp', '--prev_model_path', default='runs/origin_or_M2OR_Pairs_MolOR_650m_random_seed42_unweighted')
    parser.add_argument('-prot', '--prot_dim', type=int, default=1280)
    parser.add_argument('-esm', '--esm_version', default='650m')
    parser.add_argument('-nw', '--num_workers', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--load_cache', type=str2bool, default=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seq_cache_pt', default='runs/or_logits_shards/m2or_1237_esm650m_seq_cache.pt')
    parser.add_argument('--or_start', type=int, default=0)
    parser.add_argument('--or_end', type=int, default=1237)
    parser.add_argument('--output_pt', default='runs/or_logits_shards/shard.pt')
    parser.add_argument('--shard_glob', default='runs/or_logits_shards/shard_*.pt')
    parser.add_argument('--print_every', type=int, default=4)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.mode == 'seq_cache':
        build_or_sequence_cache(args)
    elif args.mode == 'shard':
        run_shard(args)
    elif args.mode == 'merge':
        run_merge(args)


if __name__ == '__main__':
    main()
