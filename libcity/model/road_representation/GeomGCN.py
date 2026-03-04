import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from logging import getLogger
import networkx as nx
from torch_geometric.utils import scatter, subgraph
from libcity.model.abstract_traffic_state_model import AbstractTrafficStateModel
from libcity.model import loss


class GeomGCNSingleChannel(nn.Module):
    def __init__(self, graph_data, in_feats, out_feats, num_divisions, activation, dropout_prob, merge, device):
        super(GeomGCNSingleChannel, self).__init__()
        self.num_divisions = num_divisions
        self.in_feats_dropout = nn.Dropout(dropout_prob)
        self.device = device
        self.edge_index = graph_data['edge_index']
        self.edge_subgraph_idx = graph_data['edge_subgraph_idx']
        self.norm = graph_data['norm']
        self.num_nodes = graph_data['num_nodes']

        self.linear_for_each_division = nn.ModuleList().to(self.device)
        for i in range(self.num_divisions):
            self.linear_for_each_division.append(nn.Linear(in_feats, out_feats, bias=False).to(self.device))

        for i in range(self.num_divisions):
            nn.init.xavier_uniform_(self.linear_for_each_division[i].weight.to(self.device))

        self.activation = activation
        self.subgraph_node_list_of_list = self.get_node_subgraphs()
        self.subgraph_edge_index_list = self.get_subgraphs(self.subgraph_node_list_of_list)
        self.merge = merge
        self.out_feats = out_feats

    def get_node_subgraphs(self):
        subgraph_node_list = [[] for _ in range(self.num_divisions)]
        src_list = self.edge_index[0].detach().cpu().tolist()
        dst_list = self.edge_index[1].detach().cpu().tolist()
        edge_subgraph_idx_list = self.edge_subgraph_idx.detach().cpu().tolist()
        for src, dst, edge_subgraph_idx in zip(src_list, dst_list, edge_subgraph_idx_list):
            if 0 <= edge_subgraph_idx < self.num_divisions:
                subgraph_node_list[edge_subgraph_idx].append(src)
                subgraph_node_list[edge_subgraph_idx].append(dst)

        return [
            torch.tensor(sorted(set(node_list)), dtype=torch.long, device=self.device)
            if len(node_list) > 0 else torch.empty(0, dtype=torch.long, device=self.device)
            for node_list in subgraph_node_list
        ]

    def get_subgraphs(self, subgraph_node_list_of_list):
        subgraph_edge_index_list = []
        for node_index in subgraph_node_list_of_list:
            if node_index.numel() == 0:
                subgraph_edge_index_list.append(torch.empty((2, 0), dtype=torch.long, device=self.device))
                continue
            # DGL subgraph(node_set) is replaced by PyG induced subgraph extraction.
            subgraph_edge_index, _ = subgraph(
                node_index, self.edge_index, relabel_nodes=False, num_nodes=self.num_nodes
            )
            subgraph_edge_index_list.append(subgraph_edge_index.to(self.device))
        return subgraph_edge_index_list

    def forward(self, feature):
        in_feats_dropout = self.in_feats_dropout(feature).to(self.device)

        results_from_subgraph_list = []
        for i in range(self.num_divisions):
            subgraph_edge_index = self.subgraph_edge_index_list[i]
            if subgraph_edge_index.numel() == 0:
                results_from_subgraph_list.append(
                    torch.zeros((feature.size(0), self.out_feats), dtype=in_feats_dropout.dtype, device=self.device)
                )
                continue

            # DGL copy_u + sum becomes source gather + scatter(sum) on edge_index.
            transformed_feature = self.linear_for_each_division[i](in_feats_dropout) * self.norm
            src_nodes, dst_nodes = subgraph_edge_index[0], subgraph_edge_index[1]
            aggregated_feature = scatter(
                transformed_feature[src_nodes], dst_nodes, dim=0, dim_size=self.num_nodes, reduce='sum'
            )
            results_from_subgraph_list.append(aggregated_feature)

        if self.merge == 'cat':
            h_new = torch.cat(results_from_subgraph_list, dim=-1).to(self.device)
        else:
            h_new = torch.mean(torch.stack(results_from_subgraph_list, dim=-1), dim=-1).to(self.device)
        h_new = h_new * self.norm
        h_new = self.activation(h_new)
        return h_new


class GeomGCNNet(nn.Module):
    def __init__(self, graph_data, in_feats, out_feats, num_divisions, activation, num_heads, dropout_prob, ggcn_merge,
                 channel_merge, device):
        super(GeomGCNNet, self).__init__()
        self.attention_heads = nn.ModuleList()
        for _ in range(num_heads):
            self.attention_heads.append(
                GeomGCNSingleChannel(graph_data, in_feats, out_feats, num_divisions,
                                     activation, dropout_prob, ggcn_merge, device))
        self.channel_merge = channel_merge
        self.graph_data = graph_data

    def forward(self, feature):
        all_attention_head_outputs = [head(feature) for head in self.attention_heads]
        if self.channel_merge == 'cat':
            return torch.cat(all_attention_head_outputs, dim=1)
        else:
            return torch.mean(torch.stack(all_attention_head_outputs), dim=0)


class GeomGCN(AbstractTrafficStateModel):
    def __init__(self, config, data_feature):
        super().__init__(config, data_feature)
        self.device = config.get('device')
        graph_data, num_input_features, num_output_classes, num_hidden, num_divisions, \
        num_heads_layer_one, num_heads_layer_two, dropout_rate, layer_one_ggcn_merge, \
        layer_one_channel_merge, layer_two_ggcn_merge, layer_two_channel_merge = self.get_input(config, data_feature)

        self.geomgcn1 = GeomGCNNet(graph_data, num_input_features, num_hidden, num_divisions, F.relu, num_heads_layer_one,
                                   dropout_rate, layer_one_ggcn_merge, layer_one_channel_merge, self.device)

        if layer_one_ggcn_merge == 'cat':
            layer_one_ggcn_merge_multiplier = num_divisions
        else:
            layer_one_ggcn_merge_multiplier = 1

        if layer_one_channel_merge == 'cat':
            layer_one_channel_merge_multiplier = num_heads_layer_one
        else:
            layer_one_channel_merge_multiplier = 1

        self.geomgcn2 = GeomGCNNet(graph_data,
                                   num_hidden * layer_one_ggcn_merge_multiplier * layer_one_channel_merge_multiplier,
                                   num_output_classes, num_divisions, lambda x: x,
                                   num_heads_layer_two, dropout_rate, layer_two_ggcn_merge,
                                   layer_two_channel_merge, self.device)

        self.geomgcn3 = GeomGCNNet(graph_data, num_output_classes,
                                   num_hidden * layer_one_ggcn_merge_multiplier * layer_one_channel_merge_multiplier,
                                   num_divisions, lambda x: x,
                                   num_heads_layer_two, dropout_rate, layer_two_ggcn_merge,
                                   layer_two_channel_merge, self.device)

        self.geomgcn4 = GeomGCNNet(graph_data,
                                   num_hidden * layer_one_ggcn_merge_multiplier * layer_one_channel_merge_multiplier,
                                   num_input_features,  num_divisions, F.relu, num_heads_layer_one,
                                   dropout_rate, layer_two_ggcn_merge, layer_two_channel_merge, self.device)

        self.g = graph_data
        self._logger = getLogger()
        self._scaler = self.data_feature.get('scaler')
        self.model = config.get('model', '')
        self.dataset = config.get('dataset', '')
        self.output_dim = config.get('output_dim', 8)

    def get_input(self, config, data_feature):
        num_input_features = data_feature.get('feature_dim', 1)
        num_output_classes = config.get('output_dim', 8)
        num_hidden = config.get('hidden_dim', 144)
        num_divisions = config.get('divisions_dim', 2)
        num_heads_layer_one = config.get('num_heads_layer_one', 1)
        num_heads_layer_two = config.get('num_heads_layer_two', 1)
        dropout_rate = config.get('dropout_rate', 0.5)
        layer_one_ggcn_merge = config.get('layer_one_ggcn_merge', 'cat')
        layer_two_ggcn_merge = config.get('layer_two_ggcn_merge', 'mean')
        layer_one_channel_merge = config.get('layer_one_channel_merge', 'cat')
        layer_two_channel_merge = config.get('layer_two_channel_merge', 'mean')
        adj_mx = data_feature.get('adj_mx')

        G = nx.DiGraph(adj_mx)

        for node1, node2 in list(G.edges()):
            G[node1][node2]['subgraph_idx'] = 0

        for node in sorted(G.nodes):
            if G.has_edge(node, node):
                G.remove_edge(node, node)
            G.add_edge(node, node, subgraph_idx=1)
        # DGLGraph is replaced by PyG-style tensor graph: edge_index + per-edge subgraph tags.
        edge_with_attr = list(G.edges(data='subgraph_idx'))
        src = [edge[0] for edge in edge_with_attr]
        dst = [edge[1] for edge in edge_with_attr]
        edge_subgraph_idx = [int(edge[2]) if edge[2] is not None else 0 for edge in edge_with_attr]
        edge_index = torch.tensor([src, dst], dtype=torch.long, device=self.device)
        edge_subgraph_idx = torch.tensor(edge_subgraph_idx, dtype=torch.long, device=self.device)

        degs = scatter(
            torch.ones(edge_index.size(1), dtype=torch.float32, device=self.device),
            edge_index[1],
            dim=0,
            dim_size=G.number_of_nodes(),
            reduce='sum'
        )
        norm = torch.pow(degs, -0.5)
        norm[torch.isinf(norm)] = 0
        graph_data = {
            'edge_index': edge_index,
            'edge_subgraph_idx': edge_subgraph_idx,
            'norm': norm.unsqueeze(1).to(self.device).requires_grad_(),
            'num_nodes': G.number_of_nodes()
        }

        return graph_data, num_input_features, num_output_classes, num_hidden, num_divisions, \
            num_heads_layer_one, num_heads_layer_two, dropout_rate, layer_one_ggcn_merge, \
            layer_one_channel_merge, layer_two_ggcn_merge, layer_two_channel_merge

    def forward(self, batch):
        """
        自回归任务

        Args:
            batch: dict, need key 'node_features' contains tensor shape=(N, feature_dim)

        Returns:
            torch.tensor: N, output_classes
        """
        inputs = batch['node_features']
        x = self.geomgcn1(inputs)
        encoder_state = self.geomgcn2(x)
        np.save('./libcity/cache/evaluate_cache/embedding_{}_{}_{}.npy'
                .format(self.model, self.dataset, self.output_dim),
                encoder_state.detach().cpu().numpy())
        x = self.geomgcn3(encoder_state)
        output = self.geomgcn4(x)
        return output

    def calculate_loss(self, batch):
        """
        Args:
            batch: dict, need key 'node_features', 'node_labels', 'mask'

        Returns:

        """
        y_true = batch['node_labels']
        y_predicted = self.predict(batch)
        mask = batch['mask']
        return loss.masked_mse_torch(y_predicted[mask], y_true[mask])

    def predict(self, batch):
        """
        Args:
            batch: dict, need key 'node_features'

        Returns:
            torch.tensor

        """
        return self.forward(batch)
