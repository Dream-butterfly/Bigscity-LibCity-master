import os
import random

import networkx as nx
import numpy as np
from gensim.models import Word2Vec

from GNNTP.data.dataset import TrafficStatePointDataset
from GNNTP.utils import get_dataset_cache_dir


class Graph():
    def __init__(self, nx_G, is_directed, p, q):
        self.G = nx_G
        self.is_directed = is_directed
        self.p = p
        self.q = q

    def node2vec_walk(self, walk_length, start_node):
        G = self.G
        alias_nodes = self.alias_nodes
        alias_edges = self.alias_edges
        walk = [start_node]
        while len(walk) < walk_length:
            cur = walk[-1]
            cur_nbrs = sorted(G.neighbors(cur))
            if len(cur_nbrs) > 0:
                if len(walk) == 1:
                    walk.append(cur_nbrs[alias_draw(alias_nodes[cur][0], alias_nodes[cur][1])])
                else:
                    prev = walk[-2]
                    next = cur_nbrs[alias_draw(alias_edges[(prev, cur)][0],
                                               alias_edges[(prev, cur)][1])]
                    walk.append(next)
            else:
                break
        return walk

    def simulate_walks(self, num_walks, walk_length):
        G = self.G
        walks = []
        nodes = list(G.nodes())
        print('Walk iteration:')
        for walk_iter in range(num_walks):
            print(str(walk_iter + 1), '/', str(num_walks))
            random.shuffle(nodes)
            for node in nodes:
                walks.append(self.node2vec_walk(walk_length=walk_length, start_node=node))
        return walks

    def get_alias_edge(self, src, dst):
        G = self.G
        p = self.p
        q = self.q
        unnormalized_probs = []
        for dst_nbr in sorted(G.neighbors(dst)):
            if dst_nbr == src:
                unnormalized_probs.append(G[dst][dst_nbr]['weight'] / p)
            elif G.has_edge(dst_nbr, src):
                unnormalized_probs.append(G[dst][dst_nbr]['weight'])
            else:
                unnormalized_probs.append(G[dst][dst_nbr]['weight'] / q)
        norm_const = sum(unnormalized_probs)
        normalized_probs = [float(u_prob) / norm_const for u_prob in unnormalized_probs]
        return alias_setup(normalized_probs)

    def preprocess_transition_probs(self):
        G = self.G
        is_directed = self.is_directed
        alias_nodes = {}
        for node in G.nodes():
            unnormalized_probs = [G[node][nbr]['weight'] for nbr in sorted(G.neighbors(node))]
            norm_const = sum(unnormalized_probs)
            normalized_probs = [float(u_prob) / norm_const for u_prob in unnormalized_probs]
            alias_nodes[node] = alias_setup(normalized_probs)
        alias_edges = {}
        if is_directed:
            for edge in G.edges():
                alias_edges[edge] = self.get_alias_edge(edge[0], edge[1])
        else:
            for edge in G.edges():
                alias_edges[edge] = self.get_alias_edge(edge[0], edge[1])
                alias_edges[(edge[1], edge[0])] = self.get_alias_edge(edge[1], edge[0])
        self.alias_nodes = alias_nodes
        self.alias_edges = alias_edges
        return


def alias_setup(probs):
    K = len(probs)
    q = np.zeros(K)
    J = np.zeros(K, dtype=np.int_)
    smaller = []
    larger = []
    for kk, prob in enumerate(probs):
        q[kk] = K * prob
        if q[kk] < 1.0:
            smaller.append(kk)
        else:
            larger.append(kk)
    while len(smaller) > 0 and len(larger) > 0:
        small = smaller.pop()
        large = larger.pop()
        J[small] = large
        q[large] = q[large] + q[small] - 1.0
        if q[large] < 1.0:
            smaller.append(large)
        else:
            larger.append(large)
    return J, q


def alias_draw(J, q):
    K = len(J)
    kk = int(np.floor(np.random.rand() * K))
    if np.random.rand() < q[kk]:
        return kk
    else:
        return J[kk]


def learn_embeddings(walks, dimensions, window_size, iter):
    walks = [list(map(str, walk)) for walk in walks]
    model = Word2Vec(
        walks, vector_size=dimensions, window=window_size, min_count=0, sg=1,
        workers=8, epochs=iter)
    return model


class GMANDataset(TrafficStatePointDataset):
    """GMAN 专用数据集，通过 Node2Vec 预计算空间嵌入 (SE)。

    SE 是 GMAN 的核心输入之一，在 Spatio-Temporal Embedding 中与时间嵌入结合。
    使用 Node2Vec (gensim Word2Vec) 在邻接图上游走生成节点嵌入。

    Migrated from Bigscity-LibCity-master/libcity/data/dataset/dataset_subclass/gman_dataset.py
    """

    def __init__(self, config):
        super().__init__(config)
        self.D = self.config.get('D', 64)
        self.points_per_hour = 3600 // self.time_intervals
        self.add_day_in_week = self.config.get('add_day_in_week', False)
        self.SE_config = {
            'is_directed': True, 'p': 2, 'q': 1, 'num_walks': 100,
            'walk_length': 80, 'dimensions': self.D, 'window_size': 10,
            'iter': 1000,
        }
        self.SE_config_str = (
            'SE_' + str(self.SE_config['is_directed']) + '_' + str(self.SE_config['p'])
            + '_' + str(self.SE_config['q']) + '_' + str(self.SE_config['num_walks'])
            + '_' + str(self.SE_config['walk_length']) + '_' + str(self.SE_config['dimensions'])
            + '_' + str(self.SE_config['window_size']) + '_' + str(self.SE_config['iter'])
        )
        self.SE_cache_file = os.path.join(
            get_dataset_cache_dir(),
            'SE_based_{}.txt'.format(str(self.dataset) + '_' + self.SE_config_str),
        )
        self._generate_SE()

    def _generate_SE(self):
        if not os.path.exists(self.SE_cache_file):
            nx_G = nx.from_numpy_array(self.adj_mx, create_using=nx.DiGraph())
            G = Graph(nx_G, self.SE_config['is_directed'], self.SE_config['p'], self.SE_config['q'])
            G.preprocess_transition_probs()
            walks = G.simulate_walks(self.SE_config['num_walks'], self.SE_config['walk_length'])
            model = learn_embeddings(
                walks, self.SE_config['dimensions'],
                self.SE_config['window_size'], self.SE_config['iter'],
            )
            model.wv.save_word2vec_format(self.SE_cache_file)
        SE = np.zeros(shape=(self.num_nodes, self.SE_config['dimensions']), dtype=np.float32)
        f = open(self.SE_cache_file, mode='r')
        lines = f.readlines()
        f.close()
        for line in lines[1:]:
            temp = line.split(' ')
            index = int(temp[0])
            SE[index] = temp[1:]
        print(SE.shape)
        self.SE = SE

    def get_data_feature(self):
        data_feature = super().get_data_feature()
        data_feature['SE'] = self.SE
        data_feature['D'] = self.D
        data_feature['points_per_hour'] = self.points_per_hour
        data_feature['add_day_in_week'] = self.add_day_in_week
        return data_feature
