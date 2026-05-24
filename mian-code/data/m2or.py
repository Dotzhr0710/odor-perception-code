



import pandas as pd
import numpy as np
from dgl.data.utils import get_download_dir, download, _get_dgl_url

from dgllife.data.csv_dataset import MoleculeCSVDataset
import torch
from utils import ROOT_DIR
import os


__all__ = ['Tox21']
esm_model = None
esm_alphabet = None


class M2OR(MoleculeCSVDataset):

    def __init__(self, smiles_to_graph=None,
                 node_featurizer=None,
                 edge_featurizer=None,
                 preprocess = 'original',
                 load=False,
                 log_every=1000,
                 cache_file_path='./m2or_dglgraph.bin',
                 n_jobs=1):


        if preprocess == 'original':
            data_path = 'data/datasets/M2OR_OR_odorant_pairwise_no_mixtures_dgl.csv'
        elif preprocess == 'two_class':
            data_path = 'data/datasets/M2OR_OR_odorant_pairwise_no_mixtures_dgl_two_class.csv'
        elif preprocess == 'filtered':
            data_path = 'data/datasets/M2OR_OR_odorant_pairwise_no_mixtures_dgl_filtered.csv'
        elif preprocess == 'uniprot':
            data_path = 'data/datasets/M2OR_OR_odorant_pairwise_Uniprot_no_mixtures_dgl.csv'
        elif preprocess == 'mol_OR_pairs':

            data_path = 'data/datasets/pairwise_original_m2or.csv'
        else:
            raise ValueError('Expect preprocess to be original, filtered, or two_class, got {}'.format(preprocess))

        df = pd.read_csv(data_path)

        if preprocess != 'mol_OR_pairs' and preprocess != 'two_class':
            self.id = df['InChi Key']
            df = df.drop(columns=['InChi Key'])
        else:
            self.id = df['smiles']

        self.load_full = False

        super(M2OR, self).__init__(df, smiles_to_graph, node_featurizer, edge_featurizer,
                                    "smiles", cache_file_path,
                                    load=load, log_every=log_every, n_jobs=n_jobs)

        self.id = [self.id[i] for i in self.valid_ids]

    def __getitem__(self, item):

        if self.load_full:
            return self.smiles[item], self.graphs[item], self.labels[item], \
                   self.mask[item], self.id[item]
        else:
            return self.smiles[item], self.graphs[item], self.labels[item], self.mask[item]


class M2OR_Pairs(MoleculeCSVDataset):

    def __init__(self, smiles_to_graph=None,
                 node_featurizer=None,
                 edge_featurizer=None,
                 load=False,
                 weighted_samples = False,
                 cross_attention = False,
                 esm_model = '650m',
                 esm_random_weights = False,
                 load_full = False,
                 log_every=1000,
                 cache_file_path='./m2or_dglgraph.bin',
                 max_node_len = 22,
                 n_jobs=1):
        if weighted_samples:
            data_path = 'data/datasets/M2OR_sample_weights_pairs.csv'
        else:
            data_path = 'data/datasets/M2OR_original_mol_OR_pairs.csv'

            data_path = ROOT_DIR + '/' + data_path
        df = pd.read_csv(data_path, sep=';')

        self.id =  df['mol_id'].astype(str) + '-' + df['seq_id'].astype(str)
        self.seq_id = df['seq_id'].astype(str)



        self.sequences_dict = df.groupby('seq_id').apply(lambda x: x['mutated_Sequence'].unique()).apply(pd.Series).to_dict()[0]
        sequences = list(self.sequences_dict.values())
        self.cross_attention = cross_attention
        seq_lst =  df['mutated_Sequence'].tolist()
        self.max_seq_len = len(max(sequences, key=len))
        self.seq_mask = torch.zeros((len(df), self.max_seq_len))
        for i in range(len(df)):
            self.seq_mask[i, :len(seq_lst[i])] = 1
        if cross_attention:

            for i in range(len(sequences)):
                sequences[i] += "<pad>"*(self.max_seq_len - len(sequences[i]))

            path = 'data/datasets/{}_per_residue_seq_embeddings.pth'.format(esm_model) if not esm_random_weights else '{}_per_residue_seq_embeddings_random.pth'.format(esm_model)
            if os.path.exists(path):
                seq_embeddings = torch.load(path)
            else:
                seq_embeddings = esm_embed(sequences, per_residue=True, random_weights=esm_random_weights, esm_model_version = esm_model)
                torch.save(seq_embeddings, path)

        else:
            path = 'data/datasets/{}_seq_embeddings.pth'.format(esm_model) if not esm_random_weights else '{}_seq_embeddings_random.pth'.format(esm_model)
            if os.path.exists(path):
                seq_embeddings = torch.load(path)
            else:
                seq_embeddings = esm_embed(sequences, random_weights=esm_random_weights, esm_model_version = esm_model)
                torch.save(seq_embeddings, path)
        self.seq_embeddings_dict = dict(zip(self.sequences_dict.keys(), seq_embeddings))
        if weighted_samples:
            self.sample_weights = torch.tensor(df['sample_weight'].astype(float))
            df = df.drop(columns={'weight_pair_imbalance', 'weight_class', 'weight_quality', 'sample_weight'})
        else:
            import numpy as np
            self.sample_weights = torch.tensor(pd.Series(1.0, index=np.arange(len(df)), name='orders'))

        df['smiles'] = df['canonicalSMILES']
        df = df.drop(columns={'mol_id', 'seq_id', '_DataQuality', 'num_unique_value_screen', 'mutated_Sequence'})
        df = df[['smiles', 'Responsive']]

        self.load_full = load_full


        super(M2OR_Pairs, self).__init__(df, smiles_to_graph, node_featurizer, edge_featurizer,
                                    "smiles", cache_file_path,
                                    load=load, log_every=log_every, n_jobs=n_jobs)

        self.id = [self.id[i] for i in self.valid_ids]


        if max_node_len == 22:
            self.max_node_len = max([g.number_of_nodes() for g in self.graphs])
        else:
            self.max_node_len = max_node_len
        self.graph_mask = torch.zeros((len(self.graphs), self.max_node_len))
        for idx in range(len(self.graphs)):
            self.graph_mask[idx, :self.graphs[idx].num_nodes()] = 1

    def __getitem__(self, item):

        if self.load_full:
            if self.cross_attention:
                return self.smiles[item], self.graphs[item], self.labels[item], \
                   self.mask[item], self.id[item], self.seq_id[item], \
                   self.sequences_dict[self.seq_id[item]], \
                   self.seq_embeddings_dict[self.seq_id[item]], \
                   self.sample_weights[item], self.seq_mask[item], self.graph_mask[item]
            else:
                return self.smiles[item], self.graphs[item], self.labels[item], \
                   self.mask[item], self.id[item], self.seq_id[item], self.sequences_dict[self.seq_id[item]], self.seq_embeddings_dict[self.seq_id[item]], self.sample_weights[item]
        else:
            return self.smiles[item], self.graphs[item], self.labels[item], self.mask[item]


def setup_esm(device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'), random_weights=False, esm_model_version = '650m'):
    import esm

    global esm_model
    global esm_alphabet
    if esm_model is None:
        print('loading esm model...')
        if esm_model_version == '650m':
            print('loading ESM 650M model')
            esm_model, esm_alphabet = esm.pretrained.esm2_t33_650M_UR50D()
        else:
            print('loading ESM 3B model')
            esm_model, esm_alphabet = esm.pretrained.esm2_t36_3B_UR50D()
        esm_model.eval()
        esm_model.to(device)
        print('done loading esm model')

    if random_weights:
        reinitialize_weights(esm_model)
    return esm_model, esm_alphabet

def reinitialize_weights(model):
    import torch.nn.init as init
    seed = 42
    torch.manual_seed(seed)
    for name, param in model.named_parameters():
        if 'embed' in name:
            if len(param.size()) > 1:
                init.normal_(param.data, mean=0.0, std=0.02)
        elif 'weight' in name:
            if len(param.size()) > 1:
                init.xavier_normal_(param.data)
        elif 'bias' in name:
            init.constant_(param.data, 0.0)


def esm_embed(sequences, device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'), per_residue = False, random_weights = False, esm_model_version = '650m'):

    assert isinstance(sequences, list)
    esm_model, esm_alphabet = setup_esm(random_weights=random_weights, esm_model_version = esm_model_version)
    batch_converter = esm_alphabet.get_batch_converter()

    def divide_chunks(l, n):
        for i in range(0, len(l), n):
            yield l[i:i + n]

    sequence_representations = []
    sequence_chunks = list(divide_chunks(sequences, 5))
    for sequence_chunk in sequence_chunks:
        data = []
        for i, sequence in enumerate(sequence_chunk):
            assert ' ' not in sequence

            if len(sequence) > 1600 and '<pad>' not in sequence:
                print('trimming sequence to 1600 amino acids max')
                sequence = sequence[0:1600]
            data.append(('protein' + str(i), sequence))
        batch_labels, batch_strs, batch_tokens = batch_converter(data)
        batch_lens = (batch_tokens != esm_alphabet.padding_idx).sum(1)

        layer_extracted = 33 if esm_model_version == '650m' else 36

        with torch.inference_mode():
            results = esm_model(batch_tokens.to(device), repr_layers=[layer_extracted], return_contacts=False)
        token_representations = results["representations"][layer_extracted]





        if per_residue:
            for i, tokens_len in enumerate(batch_lens):
                sequence_representations.append(token_representations[i, 1 : -1].detach().cpu().numpy())
        else:
            for i, tokens_len in enumerate(batch_lens):
                sequence_representations.append(token_representations[i, 1 : tokens_len - 1].mean(0).detach().cpu().numpy())
    print('done embedding sequences')
    return sequence_representations

def esm_embed_2(sequences, device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'), per_residue = False, random_weights = False, esm_model_version = '650m'):

    assert isinstance(sequences, list)
    esm_model, esm_alphabet = setup_esm(random_weights=random_weights, esm_model_version = esm_model_version)
    batch_converter = esm_alphabet.get_batch_converter()
    max_seq_len = 705
    def divide_chunks(l, n):
        for i in range(0, len(l), n):
            yield l[i:i + n]
    if esm_model_version == '650m':
        dim = 1280
    else:
        dim = 2560
    if per_residue:
        sequence_representations = torch.zeros(len(sequences), max_seq_len, dim)
    else:
        sequence_representations = torch.zeros(len(sequences), dim)
    sequence_chunks = list(divide_chunks(sequences, 5))
    count = 0
    for sequence_chunk in sequence_chunks:
        data = []
        for i, sequence in enumerate(sequence_chunk):
            assert ' ' not in sequence

            if len(sequence) > 1600 and '<pad>' not in sequence:
                print('trimming sequence to 1600 amino acids max')
                sequence = sequence[0:1600]
            data.append(('protein' + str(i), sequence))
        batch_labels, batch_strs, batch_tokens = batch_converter(data)
        batch_lens = (batch_tokens != esm_alphabet.padding_idx).sum(1)


        with torch.inference_mode():
            results = esm_model(batch_tokens.to(device), repr_layers=[33], return_contacts=False)
        token_representations = results["representations"][33]





        if per_residue:
            for i, tokens_len in enumerate(batch_lens):
                print(len(sequence_chunk[i]))
                print(token_representations[i, 1 : -1].shape)
                print(sequence_representations[count].shape)
                sequence_representations[count] = token_representations[i, 1 : -1].detach()

                count +=1
        else:
            for i, tokens_len in enumerate(batch_lens):
                sequence_representations[count] = token_representations[i, 1 : tokens_len - 1].mean(0).detach()
                count +=1

    print('done embedding sequences')
    return sequence_representations



def get_weight_cols(df):





    pos_weight = df['Responsive'].value_counts()[0] / df['Responsive'].value_counts()[1]
    import numpy as np
    k = 50

    for i in range(len(df)):
        if df.loc [i, '_DataQuality'] == 'ec50':
            df.loc[i, 'weight_quality'] = 1
        elif df.loc[i, '_DataQuality'] == 'primaryScreening':
            if df.loc[i, 'Responsive'] == 1:
                df.loc[i, 'weight_quality'] = 0.4
            else:
                df.loc[i, 'weight_quality'] = 0.69
        elif df.loc[i, '_DataQuality'] == 'secondaryScreening':
            if df.loc[i, 'Responsive'] == 1:
                df.loc[i, 'weight_quality'] = 0.72
            else:
                df.loc[i, 'weight_quality'] = 0.77

        if df.loc [i, 'Responsive'] == 1:
            df.loc[i, 'weight_class'] = pos_weight
        else:
            df.loc[i, 'weight_class'] = 1

        curr_receptor = df.iloc[i]['seq_id']
        curr_mol = df.iloc[i]['mol_id']
        num_mols = df[df['seq_id'] == curr_receptor]['mol_id'].shape[0]
        num_receptors = df[df['mol_id'] == curr_mol]['seq_id'].shape[0]

        df.loc[i, 'weight_pair_imbalance'] = np.log(1 + k/2 * (1/num_mols + 1/num_receptors))

    df['weight_pair_imbalance'] = df['weight_pair_imbalance'].astype(float)

    df.loc[i, 'sample_weight'] = df.iloc[i]['weight_quality'] * df.iloc[i]['weight_class'] * df.iloc[i]['weight_pair_imbalance']

    return df



class GS_LF(MoleculeCSVDataset):

    def __init__(self, smiles_to_graph=None,
                 node_featurizer=None,
                 edge_featurizer=None,
                 smiles_type = 'canonical',
                 load=False,
                 log_every=1000,
                 cache_file_path='./gs_lf_dglgraph.bin',
                 n_jobs=1):
        from rdkit import Chem






        data_path = 'data/datasets/NaNs_GS_LF_isomeric_SMILES_dedup_odor_filtered.csv'
        df = pd.read_csv(data_path)
        self.id = df['CID']

        df = df.drop(columns=['Stimulus', 'CID', 'IUPACName', 'MolecularWeight', 'name'])
        if smiles_type == 'canonical':
            iso_smiles = df['IsomericSMILES'].tolist()
            df['smiles'] = [Chem.MolToSmiles(Chem.MolFromSmiles(smiles)) for smiles in iso_smiles]

            df = df[['smiles'] + [col for col in df.columns if col != 'smiles']]
            df = df.drop(columns=['IsomericSMILES'])
        self.load_full = False

        super(GS_LF, self).__init__(df, smiles_to_graph, node_featurizer, edge_featurizer,
                                    "smiles", cache_file_path,
                                    load=load, log_every=log_every, n_jobs=n_jobs)

        self.id = [self.id[i] for i in self.valid_ids]

    def __getitem__(self, item):

        if self.load_full:
            return self.smiles[item], self.graphs[item], self.labels[item], \
                   self.mask[item], self.id[item]
        else:
            return self.smiles[item], self.graphs[item], self.labels[item], self.mask[item]


class GS_LF_OR(MoleculeCSVDataset):

    def __init__(self, smiles_to_graph=None,
                 node_featurizer=None,
                 edge_featurizer=None,
                 smiles_type = 'canonical',
                 load=False,
                 log_every=1000,
                 cache_file_path='./gs_lf_dglgraph.bin',
                 n_jobs=1):
        from rdkit import Chem
















        data_path = 'data/datasets/NaNs_GS_LF_isomeric_SMILES_dedup_odor_filtered.csv'
        df = pd.read_csv(data_path)
        self.id = df['CID']

        df = df.drop(columns=['Stimulus', 'CID', 'IUPACName', 'MolecularWeight', 'name'])
        if smiles_type == 'canonical':
            iso_smiles = df['IsomericSMILES'].tolist()
            df['smiles'] = [Chem.MolToSmiles(Chem.MolFromSmiles(smiles)) for smiles in iso_smiles]

            df = df[['smiles'] + [col for col in df.columns if col != 'smiles']]
            df = df.drop(columns=['IsomericSMILES'])
        self.load_full = load

        super(GS_LF_OR, self).__init__(df, smiles_to_graph, node_featurizer, edge_featurizer,
                                    "smiles", cache_file_path,
                                    load=load, log_every=log_every, n_jobs=n_jobs)

        self.id = [self.id[i] for i in self.valid_ids]


        self.max_node_len = max([g.number_of_nodes() for g in self.graphs])
        self.graph_mask = torch.zeros((len(self.graphs), self.max_node_len))
        for idx in range(len(self.graphs)):
            self.graph_mask[idx, :self.graphs[idx].num_nodes()] = 1


    def __getitem__(self, item):

        if self.load_full:
            return item, self.smiles[item], self.graphs[item], self.labels[item], \
                   self.mask[item], self.id[item], self.graph_mask[item]
        else:
            return item, self.smiles[item], self.graphs[item], self.labels[item], self.mask[item]


