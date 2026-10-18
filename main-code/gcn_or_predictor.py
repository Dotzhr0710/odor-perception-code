import torch
import torch.nn as nn

from dgllife.model.model_zoo.mlp_predictor import MLPPredictor
from dgllife.model.gnn.gcn import GCN
from dgllife.model.gnn.gatv2 import GATv2
from dgllife.model.readout.weighted_sum_and_max import WeightedSumAndMax
from torch.nn.functional import scaled_dot_product_attention
import dgl
import numpy as np

class GCNJointPredictor(nn.Module):


    def __init__(self, in_feats, hidden_feats=None, gnn_norm=None, activation=None,
                 add_feats = False,
                 residual=None, batchnorm=None, dropout=None, classifier_hidden_feats=128,
                 classifier_dropout=0., n_tasks=[574, 152], predictor_hidden_feats=128,
                 predictor_dropout=0.):
        super(GCNJointPredictor, self).__init__()

        if predictor_hidden_feats == 128 and classifier_hidden_feats != 128:
            print('classifier_hidden_feats is deprecated and will be removed in the future, '
                  'use predictor_hidden_feats instead')
            predictor_hidden_feats = classifier_hidden_feats

        if predictor_dropout == 0. and classifier_dropout != 0.:
            print('classifier_dropout is deprecated and will be removed in the future, '
                  'use predictor_dropout instead')
            predictor_dropout = classifier_dropout

        self.gnn = GCN(in_feats=in_feats,
                       hidden_feats=hidden_feats,
                       gnn_norm=gnn_norm,
                       activation=activation,
                       residual=residual,
                       batchnorm=batchnorm,
                       dropout=dropout)
        gnn_out_feats = self.gnn.hidden_feats[-1]
        self.add_feats = add_feats
        self.readout = WeightedSumAndMax(gnn_out_feats)

        self.predict_ORs = MLPPredictor(2 * gnn_out_feats, predictor_hidden_feats,
                                        n_tasks[0], predictor_dropout)

        if add_feats:
            self.predict_scent = MLPPredictor(2 * gnn_out_feats + n_tasks[0], predictor_hidden_feats,
                n_tasks[1], predictor_dropout)
        else:
            self.predict_scent = MLPPredictor(2 * gnn_out_feats, predictor_hidden_feats,
                                    n_tasks[1], predictor_dropout)
    def forward(self, bg, feats):

        node_feats = self.gnn(bg, feats)
        graph_feats = self.readout(bg, node_feats)

        OR_logits = self.predict_ORs(graph_feats)
        if self.add_feats is True:
            graph_feats_scent = torch.cat((graph_feats, OR_logits), dim=1)
        else:
            graph_feats_scent = graph_feats

        return OR_logits, self.predict_scent(graph_feats_scent)



class GCNORPredictor(nn.Module):

    def __init__(self, in_feats, hidden_feats=None, gnn_norm=None, activation=None,
                 add_feats = None,
                 residual=None, batchnorm=None, dropout=None, classifier_hidden_feats=128,
                 classifier_dropout=0., n_tasks=1, predictor_hidden_feats=128,
                 predictor_dropout=0.):
        super(GCNORPredictor, self).__init__()

        if predictor_hidden_feats == 128 and classifier_hidden_feats != 128:
            print('classifier_hidden_feats is deprecated and will be removed in the future, '
                  'use predictor_hidden_feats instead')
            predictor_hidden_feats = classifier_hidden_feats

        if predictor_dropout == 0. and classifier_dropout != 0.:
            print('classifier_dropout is deprecated and will be removed in the future, '
                  'use predictor_dropout instead')
            predictor_dropout = classifier_dropout

        self.gnn = GCN(in_feats=in_feats,
                       hidden_feats=hidden_feats,
                       gnn_norm=gnn_norm,
                       activation=activation,
                       residual=residual,
                       batchnorm=batchnorm,
                       dropout=dropout)
        gnn_out_feats = self.gnn.hidden_feats[-1]
        self.readout = WeightedSumAndMax(gnn_out_feats)
        if add_feats:
            self.predict = MLPPredictor(2 * gnn_out_feats + add_feats, predictor_hidden_feats,
                                    n_tasks, predictor_dropout)
        else:
            self.predict = MLPPredictor(2 * gnn_out_feats, predictor_hidden_feats,
                                    n_tasks, predictor_dropout)

    def forward(self, bg, feats, add_feats = None):

        node_feats = self.gnn(bg, feats)
        graph_feats = self.readout(bg, node_feats)
        if add_feats.dim() > 2:
            add_feats = add_feats.squeeze(1)


        if add_feats is not None:
            graph_feats = torch.cat((graph_feats, add_feats), dim=1)

        return self.predict(graph_feats)






class CrossAttention(nn.Module):



    def __init__(self, D1, D2, mol2prot = False):
        super(CrossAttention, self).__init__()


        if mol2prot:
            self.query_transform_tensor1 = nn.Linear(D1, D1)
            self.key_transform_tensor1 = nn.Linear(D1, D1)
            self.value_transform_tensor1 = nn.Linear(D1, D1)

            self.query_transform_tensor2 = nn.Linear(D2, D1)
            self.key_transform_tensor2 = nn.Linear(D2, D1)
            self.value_transform_tensor2 = nn.Linear(D2, D1)


            self.linear1 = nn.Linear(D1, 1)
            self.linear2 = nn.Linear(D1, 1)
        else:
            self.query_transform_tensor1 = nn.Linear(D1, D2)
            self.key_transform_tensor1 = nn.Linear(D1, D2)
            self.value_transform_tensor1 = nn.Linear(D1, D2)

            self.query_transform_tensor2 = nn.Linear(D2, D2)
            self.key_transform_tensor2 = nn.Linear(D2, D2)
            self.value_transform_tensor2 = nn.Linear(D2, D2)


            self.linear1 = nn.Linear(D2, 1)
            self.linear2 = nn.Linear(D2, 1)

    def scaled_attention_weights(self, query, key, value):
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / torch.sqrt(torch.tensor(d_k).float())

        attention_weights = torch.relu(scores)

        return attention_weights

    def gen_attn_maps(self, tensor1, tensor2, seq_mask, node_mask):

        tensor1 = tensor1 * seq_mask[:, :, np.newaxis]




        query_tensor1 = self.query_transform_tensor1(tensor1)
        key_tensor1 = self.key_transform_tensor1(tensor1)
        value_tensor1 = self.value_transform_tensor1(tensor1)

        query_tensor2 = self.query_transform_tensor2(tensor2)
        key_tensor2 = self.key_transform_tensor2(tensor2)
        value_tensor2 = self.value_transform_tensor2(tensor2)

        prot_attention_maps = self.scaled_attention_weights(query_tensor1, key_tensor2, value_tensor2)
        mol_attention_maps = self.scaled_attention_weights(query_tensor2, key_tensor1, value_tensor1)

        return prot_attention_maps, mol_attention_maps

    def forward(self, tensor1, tensor2, seq_mask, node_mask):


        tensor1 = tensor1 * seq_mask[:, :, np.newaxis]




        query_tensor1 = self.query_transform_tensor1(tensor1)
        key_tensor1 = self.key_transform_tensor1(tensor1)
        value_tensor1 = self.value_transform_tensor1(tensor1)

        query_tensor2 = self.query_transform_tensor2(tensor2)
        key_tensor2 = self.key_transform_tensor2(tensor2)
        value_tensor2 = self.value_transform_tensor2(tensor2)




        attended_values_tensor1 = scaled_dot_product_attention(query_tensor1, key_tensor2, value_tensor2)
        attended_values_tensor2 = scaled_dot_product_attention(query_tensor2, key_tensor1, value_tensor1)






        fixed_size_tensor1 = self.linear1(attended_values_tensor1).squeeze(-1)
        fixed_size_tensor2 = self.linear2(attended_values_tensor2).squeeze(-1)





        fixed_size_tensor1[seq_mask == 0] = 0
        fixed_size_tensor2[node_mask == 0] = 0





        tensor1 = tensor1.transpose(1, 2)
        protein_vec = torch.einsum('ijk,ik->ij', tensor1, fixed_size_tensor1)

        tensor2 = tensor2.transpose(1, 2)
        mol_vec = torch.einsum('ijk, ik->ij', tensor2, fixed_size_tensor2)

        output_vec = torch.cat((protein_vec, mol_vec), dim=1)


        return output_vec




        output_vec = torch.cat((fixed_size_tensor1, fixed_size_tensor2), dim=1)


        return output_vec

class OdorantReceptorCrossAttention(nn.Module):



    def __init__(self, D1, D2, mol2prot = False):
        super(OdorantReceptorCrossAttention, self).__init__()


        if mol2prot:
            self.query_transform_tensor1 = nn.Linear(D1, D1)
            self.key_transform_tensor1 = nn.Linear(D1, D1)
            self.value_transform_tensor1 = nn.Linear(D1, D1)

            self.query_transform_tensor2 = nn.Linear(D2, D1)
            self.key_transform_tensor2 = nn.Linear(D2, D1)
            self.value_transform_tensor2 = nn.Linear(D2, D1)


            self.linear1 = nn.Linear(D1, 1)
            self.linear2 = nn.Linear(D1, 1)
        else:
            self.query_transform_tensor1 = nn.Linear(D1, D2)
            self.key_transform_tensor1 = nn.Linear(D1, D2)
            self.value_transform_tensor1 = nn.Linear(D1, D2)

            self.query_transform_tensor2 = nn.Linear(D2, D2)
            self.key_transform_tensor2 = nn.Linear(D2, D2)
            self.value_transform_tensor2 = nn.Linear(D2, D2)


            self.linear1 = nn.Linear(D2, 1)
            self.linear2 = nn.Linear(D2, 1)

    def scaled_dot_product_attention(self, query, key, value):
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / torch.sqrt(torch.tensor(d_k).float())

        attention_weights = torch.relu(scores)

        attended_values = torch.matmul(attention_weights, value)
        return attended_values

    def scaled_attention_weights(self, query, key, value):
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / torch.sqrt(torch.tensor(d_k).float())

        attention_weights = torch.relu(scores)

        return attention_weights

    def gen_attn_maps(self, tensor1, tensor2, seq_mask, node_mask):

        tensor1 = tensor1 * seq_mask[:, :, np.newaxis]




        query_tensor1 = self.query_transform_tensor1(tensor1)
        key_tensor1 = self.key_transform_tensor1(tensor1)
        value_tensor1 = self.value_transform_tensor1(tensor1)

        query_tensor2 = self.query_transform_tensor2(tensor2)
        key_tensor2 = self.key_transform_tensor2(tensor2)
        value_tensor2 = self.value_transform_tensor2(tensor2)

        prot_attention_maps = self.scaled_attention_weights(query_tensor1, key_tensor2, value_tensor2)
        mol_attention_maps = self.scaled_attention_weights(query_tensor2, key_tensor1, value_tensor1)

        return prot_attention_maps, mol_attention_maps

    def forward(self, tensor1, tensor2, seq_mask, node_mask):


        tensor1 = tensor1 * seq_mask[:, :, np.newaxis]




        query_tensor1 = self.query_transform_tensor1(tensor1)
        key_tensor1 = self.key_transform_tensor1(tensor1)
        value_tensor1 = self.value_transform_tensor1(tensor1)

        query_tensor2 = self.query_transform_tensor2(tensor2)
        key_tensor2 = self.key_transform_tensor2(tensor2)
        value_tensor2 = self.value_transform_tensor2(tensor2)


        attended_values_tensor1 = self.scaled_dot_product_attention(query_tensor1, key_tensor2, value_tensor2)
        attended_values_tensor2 = self.scaled_dot_product_attention(query_tensor2, key_tensor1, value_tensor1)





        fixed_size_tensor1 = self.linear1(attended_values_tensor1).squeeze(-1)
        fixed_size_tensor2 = self.linear2(attended_values_tensor2).squeeze(-1)





        fixed_size_tensor1[seq_mask == 0] = 0
        fixed_size_tensor2[node_mask == 0] = 0





        tensor1 = tensor1.transpose(1, 2)
        protein_vec = torch.einsum('ijk,ik->ij', tensor1, fixed_size_tensor1)

        tensor2 = tensor2.transpose(1, 2)
        mol_vec = torch.einsum('ijk, ik->ij', tensor2, fixed_size_tensor2)

        output_vec = torch.cat((protein_vec, mol_vec), dim=1)


        return output_vec



class MolORPredictor(nn.Module):

    def __init__(self, in_feats, hidden_feats=None, gnn_norm=None, activation=None,
                 add_feats = None, prot_feats = 1280, gnn_attended_feats = None, max_seq_len = 705, max_node_len = 22,
                 mol2_prot = False,
                 residual=None, batchnorm=None, dropout=None, classifier_hidden_feats=128,
                 classifier_dropout=0., n_tasks=1, predictor_hidden_feats=128,
                 predictor_dropout=0.):
        super(MolORPredictor, self).__init__()
        torch.autograd.set_detect_anomaly(True)

        if predictor_hidden_feats == 128 and classifier_hidden_feats != 128:
            print('classifier_hidden_feats is deprecated and will be removed in the future, '
                  'use predictor_hidden_feats instead')
            predictor_hidden_feats = classifier_hidden_feats

        if predictor_dropout == 0. and classifier_dropout != 0.:
            print('classifier_dropout is deprecated and will be removed in the future, '
                  'use predictor_dropout instead')
            predictor_dropout = classifier_dropout
        self.max_node_len = max_node_len

        self.gnn = GCN(in_feats=in_feats,
                       hidden_feats=hidden_feats,
                       gnn_norm=gnn_norm,
                       activation=activation,
                       residual=residual,
                       batchnorm=batchnorm,
                       dropout=dropout)
        gnn_out_feats = self.gnn.hidden_feats[-1]




        self.cross_attn = OdorantReceptorCrossAttention(prot_feats, gnn_out_feats, mol2prot = mol2_prot)

        gnn_attended_feats = self.gnn.hidden_feats[-1] if gnn_attended_feats is None else gnn_attended_feats

        self.predict = MLPPredictor(prot_feats + gnn_attended_feats, predictor_hidden_feats,
                                    n_tasks, predictor_dropout)


        self.prot_norm = nn.LayerNorm(prot_feats)
        self.mol_norm = nn.LayerNorm(gnn_out_feats)
        self.feat_norm = nn.LayerNorm(prot_feats + gnn_attended_feats)




    def forward(self, bg, feats, add_feats = None, seq_mask = None, node_mask = None, device = None):



        node_feats = self.gnn(bg, feats)








        graphs = dgl.unbatch(bg)
        batch_node_feats = torch.zeros((len(graphs), self.max_node_len, node_feats.shape[1]))


        counter = 0
        for i in range(len(graphs)):
            n_nodes = graphs[i].num_nodes()
            batch_node_feats[i][:n_nodes] = node_feats[counter:n_nodes + counter]
            counter+=n_nodes












        if torch.cuda.is_available() and device is not None:
            add_feats = add_feats.to(device)
            batch_node_feats = batch_node_feats.to(device)











        add_feats = self.prot_norm(add_feats)
        batch_node_feats = self.mol_norm(batch_node_feats)

        graph_feats = self.cross_attn(add_feats, batch_node_feats, seq_mask, node_mask)



        graph_feats = self.feat_norm(graph_feats)

        return self.predict(graph_feats)

    def generate_attention_maps(self, bg, feats, add_feats = None, seq_mask = None, node_mask = None, device = None):
        node_feats = self.gnn(bg, feats)








        graphs = dgl.unbatch(bg)
        batch_node_feats = torch.zeros((len(graphs), self.max_node_len, node_feats.shape[1]))


        counter = 0
        for i in range(len(graphs)):
            n_nodes = graphs[i].num_nodes()
            batch_node_feats[i][:n_nodes] = node_feats[counter:n_nodes + counter]
            counter+=n_nodes


        if torch.cuda.is_available() and device is not None:
            add_feats = add_feats.to(device)
            batch_node_feats = batch_node_feats.to(device)




        add_feats = self.prot_norm(add_feats)
        batch_node_feats = self.mol_norm(batch_node_feats)

        prot_attention_maps, mol_attention_maps = self.cross_attn.gen_attn_maps(add_feats, batch_node_feats, seq_mask, node_mask)

        return prot_attention_maps, mol_attention_maps



class Mol_JointPredictor(nn.Module):

    def __init__(self, in_feats, hidden_feats=None, gnn_norm=None, activation=None,
                 add_feats = None, prot_feats = 1280, max_seq_len = 705, max_node_len = 22,
                 mol2_prot = False,
                 residual=None, batchnorm=None, dropout=None, classifier_hidden_feats=128,
                 classifier_dropout=0., n_tasks=1, predictor_hidden_feats=128,
                 predictor_dropout=0.):
        super(Mol_JointPredictor, self).__init__()

        if predictor_hidden_feats == 128 and classifier_hidden_feats != 128:
            print('classifier_hidden_feats is deprecated and will be removed in the future, '
                  'use predictor_hidden_feats instead')
            predictor_hidden_feats = classifier_hidden_feats

        if predictor_dropout == 0. and classifier_dropout != 0.:
            print('classifier_dropout is deprecated and will be removed in the future, '
                  'use predictor_dropout instead')
            predictor_dropout = classifier_dropout

        self.gnn = GCN(in_feats=in_feats,
                       hidden_feats=hidden_feats,
                       gnn_norm=gnn_norm,
                       activation=activation,
                       residual=residual,
                       batchnorm=batchnorm,
                       dropout=dropout)
        gnn_out_feats = self.gnn.hidden_feats[-1]
        self.readout = WeightedSumAndMax(gnn_out_feats)

        self.cross_attn = CrossAttention(prot_feats, gnn_out_feats, mol2prot = mol2_prot)

        self.predict_OR = MLPPredictor(max_seq_len + max_node_len, predictor_hidden_feats,
                                    n_tasks[0], predictor_dropout)

        self.predict_scent = MLPPredictor(2 * gnn_out_feats, predictor_hidden_feats,
                                    n_tasks[1], predictor_dropout)


    def forward(self, bg, feats, add_feats = None, seq_mask = None, node_mask = None):



        node_feats = self.gnn(bg, feats)
        graph_feats = self.readout(bg, node_feats)


        if add_feats is not None:





            graphs = dgl.unbatch(bg)
            batch_node_feats = torch.zeros((len(graphs), 22, node_feats.shape[1]))


            counter = 0
            for i in range(len(graphs)):
                n_nodes = graphs[i].num_nodes()
                batch_node_feats[i][:n_nodes] = node_feats[counter:n_nodes + counter]
                counter+=n_nodes











            if torch.cuda.is_available():
                add_feats = add_feats.cuda()
                batch_node_feats = batch_node_feats.cuda()






            OR_feats = self.cross_attn(add_feats, batch_node_feats, seq_mask, node_mask)


        if add_feats is None:
            return self.predict_scent(graph_feats)
        else:
            return self.predict_scent(graph_feats), self.predict_OR(OR_feats)
