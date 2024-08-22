
import pickle
import time
import math
import time
from datetime import datetime
import numpy as np
import networkx as nx

from gensim.models.word2vec import Word2Vec
# THis requires copy of original node2vec.py
try:
    from module.node2vec import *
except ModuleNotFoundError as e:
    from node2vec import *

from abc import ABC, abstractmethod

# Loosely following this example : https://stellargraph.readthedocs.io/en/stable/demos/link-prediction/node2vec-link-prediction.html#Node2Vec
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from typing import List, Tuple, Dict
from collections import Counter
try:
    from module.experiments import *
except ModuleNotFoundError as e:
    from experiments import *


SEED=42

import gensim
print(gensim.__version__)

random.seed(SEED)
np.random.seed(SEED)

def generate_node2vec_embeddings(nx_G, 
                                 emb_size=64, 
                                 num_walks=20, 
                                 walk_length=80,
                                 window=10,
                                 workers=4,
                                 algo=1,
                                 play_pq = False,
                                 directed = False):

    print("Generating graph edges.")
    ## Add weight to the edges
    for edge in nx_G.edges():
        nx_G[edge[0]][edge[1]]['weight'] = 1
        if not directed:
            nx_G[edge[1]][edge[0]]['weight'] = 1
    
    if play_pq:
        walks = []
        # Number of walks must devide by 4
        assert num_walks % 4 == 0
        pqs = [(0.5, 1.0), (1.0, 0.5), (0.25, 2.0), (2.0, 0.25)]
        p, q = pqs[0]
        print("Generating walks ...", datetime.now(), " Num walks: ", num_walks)
        G = Graph(nx_G, is_directed=directed, p=p, q=q)
        for i in range(len(pqs)):
            p, q = pqs[i]
            G.p = p
            G.q = q
            G.preprocess_transition_probs()
            wal = G.simulate_walks(num_walks=num_walks//4, walk_length=walk_length)
            walks += [list(map(str, walk)) for walk in wal]
            print("Walks ready : ", i, " / ", len(pqs), " ", datetime.now())
    else:
        print("Generating walks 2: ", num_walks)
        G = Graph(nx_G, is_directed=directed, p=1, q=1)
        G.preprocess_transition_probs()
        walks = G.simulate_walks(num_walks=num_walks, walk_length=walk_length)
        walks = [list(map(str, walk)) for walk in walks]
    print("Creating model with  Workers: ", workers)
    model = Word2Vec(walks, vector_size=emb_size, window=window, min_count=0, sg=algo, workers=workers)
    wv = model.wv
    num_nodes = len(nx_G.nodes())
    embeddings = np.zeros([num_nodes, emb_size], dtype='float32')
    sum_embeddings = 0
    empty_list = []
    # This algorithm is made for subing empty embeddings 
    # with mean embeddings of the graph. 
    for i in range(num_nodes):
        if str(i) in wv:
            embeddings[i] = wv.word_vec(str(i))
            sum_embeddings += embeddings[i]
        else:
            empty_list.append(i)
    print('Found empty embeddings.', len(empty_list))
    mean_embedding = sum_embeddings / (num_nodes - len(empty_list))
    embeddings[empty_list] = mean_embedding
    return embeddings, walks


def link_prediction_classifier(max_iter=2000):
    lr_clf = LogisticRegressionCV(Cs=10, cv=10, scoring="roc_auc", max_iter=max_iter)
    return Pipeline(steps=[("sc", StandardScaler()), ("clf", lr_clf)])

def train_link_prediction_model(labels, edge_embeddings):
    clf = link_prediction_classifier()
    clf.fit(edge_embeddings, labels)
    return clf


class MyNode2VecConfig(Configuration):
    
    def __init__(self, oper, 
                 emb_size:int = 32, 
                 num_walks: int = 10,
                 walk_size: int = 10,
                 window: int = 10,
                 workers: int = 4,
                 algo: int = 1,
                 multi_sample: int = 1,
                 play_pq: bool = False):
        self.directed = False
        self.operator = oper
        self.emb_size = emb_size
        self.num_walks = num_walks
        self.window = window
        self.play_pq = play_pq
        self.walk_size = walk_size
        self.workers = workers
        self.algo = algo
        self.multi_sample = multi_sample
    
    #override
    def isDirected(self):
        return self.directed

class MyNode2VecModel(AbsModel):
    def __init__(self):
        self.clf = None
        self.emb = None
        self.op = None
        self.data = None
        self.walks = None
    
    def train(self, config,  
              train_labels: List, 
              train_edg: List,
              real_nodes: List,
              real_edges: List):
        
        ## NODE_2_VEC SPECIFIC TRAINING
        self.data  = (train_edg, train_labels, real_nodes, real_edges)
        edges = [train_edg[i] for i in range(len(train_edg)) if train_labels[i] > 0]        
        A = get_adj(real_nodes, edges, config.isDirected)
        graph = convert_to_graph(A, config.isDirected)
        self.emb, walks = generate_node2vec_embeddings(graph, 
                                                    emb_size=config.emb_size, 
                                                    num_walks=config.num_walks, 
                                                    walk_length=config.walk_size,
                                                    window=config.window,
                                                    workers=config.workers,
                                                    algo=config.algo,
                                                    play_pq=config.play_pq,
                                                    directed=config.isDirected)
        self.walks = walks
        self.op = config.operator
        train_features = [self.op(self.emb[e[0]], self.emb[e[1]]) for e in train_edg]       
        
        assert sum(train_labels) == len(real_edges) and train_labels[0] == 0 and 1.0 - train_labels[-1] < 0.001
        assert len(train_labels) == len(real_edges)*2 and len(train_features) == len(train_labels)
        
        self.clf = train_link_prediction_model(train_labels, train_features)
    
    def predict(self, test_edg):
        test_features = [self.op(self.emb[e[0]], self.emb[e[1]]) for e in test_edg]
        predicted = self.clf.predict_proba(test_features)
        return predicted
    
    def reset(self):
        self.clf = None
        self.emb = None
        self.walks = None

def n2v_work(data: dict, params: dict):
    walk_size = params.get('walk_size', 20)
    workers = params.get('workers', 8)
    emb_size = params.get('emb_size', 128)
    ws = params.get('num_walks', 100)
    window = params.get('window', 10)
    algo = params.get('algo', 1)
    multi = params.get('multi', 1)
    print("Node2vec - Workers: ", workers, 'Walk size: ', walk_size, ' Embedding size: ', emb_size, ' Window:', window)
    config = MyNode2VecConfig(oper=operator_hadamard, 
                              emb_size=emb_size, 
                              num_walks=ws*4, 
                              window=window,
                              algo=algo,
                              workers=workers,
                              multi_sample=multi,
                              play_pq=True)
    config.isDirected = False
    config.walk_size = walk_size
    model = MyNode2VecModel()
    score, ntime = compute_file_no_sampling_results(config, model, data)
    return (score, ntime, model)

