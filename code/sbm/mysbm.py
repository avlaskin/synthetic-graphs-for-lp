import pickle
import numpy as np
import time
import re
import math
import random
import graph_tool.all as gt
import matplotlib.pyplot as plt

from collections import defaultdict
from abc import ABC, abstractmethod
from typing import List, Tuple, Dict

from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from experiments import *

_N  = 16

def plot_matrix(good_state, level=2):
    if isinstance(good_state, gt.NestedBlockState):
        levels = good_state.get_levels()
        state = levels[level]
        e = state.get_matrix()
        fig, ax = plt.subplots(figsize=(20,20))
        ax.matshow(e.todense())
    else:
        state = good_state
        e = state.get_matrix()
        fig, ax = plt.subplots(figsize=(20,20))
        ax.matshow(e.todense())
        
def get_model(g, fn_iters=20, niter=100, nested=False, deg_corr=True, verbose=True, toSweep=False):
    sequential = False
    good_state = None
    state = None
    # Deg_corr = 78674 vs 85770 vs 76003 (Nested)
    print("Model Nested: ", nested, " Deg. Correction:", deg_corr, " To sweep: ", toSweep)
    mdl = 1e+10
    for i in range(fn_iters):
        st = time.monotonic()
        if nested:
            state = gt.minimize_nested_blockmodel_dl(g)
        else:
            state = gt.minimize_blockmodel_dl(g, state_args=dict(deg_corr=deg_corr))
        # This part makes the model even better:
        if verbose:
            print("Initial entropy:", state.entropy())
        if toSweep:
            for j in range(20):
                ret = state.multiflip_mcmc_sweep(niter=niter, beta=np.inf)
                print(".", end="")
        if mdl > state.entropy():
            mdl = state.entropy()
            good_state = state
            if verbose:
                print("Found better state entropy: ", mdl)
            else:
                print("+", end="")
            print("Time took: ", time.monotonic() - st)
        else:
            if verbose:
                print("Model: ", state.entropy(), " itteration ", i)
            else:
                print("-", end="")
    print("Best Model: ", good_state.entropy(), " itteration ", i)
    return good_state


# 1. Set of functions for hashing and the test for them.
def convert2hash(x: int, y: int, n=_N):
    return (x<<n) + y

def convert2tuple(hashval, n=_N):
    return (hashval>>n, hashval & 0x0000FFFF)

def testConverters():
    for i in range(1023):
        for j in range(i+1, 1023):
            h1 = convert2hash(i, j)
            h2 = convert2hash(j, i)
            x1, y1 = convert2tuple(h1)
            x2, y2 = convert2tuple(h2)
            if x1 != i or y1 != j or x2 != j or y2 != i:
                print("Test failed", i, j)
                break
# 3. Container for all samples.
testConverters()

class EdgeProbabilities(object):
    def __init__(self, all_edges, partition_size):
        self.edge_probs = defaultdict(list)
        self.current_pos = 0
        self.partition_size = partition_size
        self.current_partition = []
        self.current_partition_probs = []
        self.edges = all_edges
        
    def prepareNext(self):
        end = self.current_pos +  self.partition_size
        end = end if end <= len(self.edges) else len(self.edges)
        self.current_partition_probs = [0]*(end - self.current_pos)
        self.current_partition = [0]*(end - self.current_pos)
        for i in range(self.current_pos, end):
             self.current_partition[i - self.current_pos] = self.edges[i]
    
    def checkNext(self):
        end = self.current_pos +  self.partition_size
        if end <= len(self.edges):
            self.current_pos += self.partition_size
            return True
        self.current_pos = 0
        return False
                
    def addProbabilities(self, probs):
        for i in range(len(self.current_partition)):
            hashval = convert2hash(self.current_partition[i][0], self.current_partition[i][1])
            if hashval == 28508829:
                print(i, self.current_partition[i][0], self.current_partition[i][1])
            self.edge_probs[hashval].append(probs[i])

def generate_undirected(sparse_adj_matrix) -> nx.Graph:
    g = gt.Graph(directed=False)
    e1, e2 = sparse_adj_matrix.nonzero()
    vs = []
    for k in range(sparse_adj_matrix.shape[0]):
        v = g.add_vertex()
        vs.append(v)
    for j, _ in enumerate(e1):
        g.add_edge(e1[j], e2[j])
    return g

class MySBMConfig(Configuration):
    def __init__(self, iterations=5, degreeCorrected=True):
        self.directed = False
        self.iterations = iterations
        self.degreeCorrected = degreeCorrected
        
    def isDirected(self):
        return self.directed

class MySBMModel(AbsModel):
    
    def __init__(self, iterations: int=5):
        self.edges = None
        self.iterations = iterations
        self.good_state = None
        self.spli = 10
        self.g = None
        self.vs = None
        self.edgeProbs = None
    
    def train(self, 
              config,  
              train_labels: list, 
              train_edg: list,
              real_nodes: list,
              real_edges: list
             ):
        self.edges = real_edges
        self.g = gt.Graph(directed=False)
        # Create Vertexes
        self.vs = []
        for i, k in enumerate(real_nodes):
            v = self.g.add_vertex()
            self.vs.append(v)
        # Create Nodes 
        for i, e in enumerate(train_edg):
            if train_labels[i]:
                self.g.add_edge(self.vs[e[0]], self.vs[e[1]])
                self.g.add_edge(self.vs[e[1]], self.vs[e[0]])
        self.g.set_directed(False)
        self.good_state = get_model(self.g, 
                                    fn_iters=config.iterations,
                                    niter=5000,
                                    nested=False, 
                                    deg_corr=config.degreeCorrected, 
                                    verbose=False)
    
    def predict(self, test_edg):
        spli = len(test_edg) // 2
        self.edgeProbs = EdgeProbabilities(test_edg, spli)
        
        def getIndexProbs(edge_idx, ep=self.edgeProbs):
            hashval = convert2hash(ep.edges[edge_idx][0], ep.edges[edge_idx][1])
            p = ep.edge_probs[hashval]
            return p
        def get_avg(p):
            p = np.array(p)
            pmax = p.max()
            p -= pmax
            return pmax + np.log(np.exp(p).mean())
        def collect_edge_probs(sampled_state, ep=self.edgeProbs):
            ep.prepareNext()
            probs = []
            for i in range(len(ep.current_partition)):
                probs.append(sampled_state.get_edges_prob([ep.current_partition[i]],
                                                         entropy_args=dict(partition_dl=False)))
            ep.addProbabilities(probs)
            while ep.checkNext():
                ep.prepareNext()
                probs = []
                for i in range(len(ep.current_partition)):
                    probs.append(sampled_state.get_edges_prob([ep.current_partition[i]],
                                                         entropy_args=dict(partition_dl=False)))
                ep.addProbabilities(probs)
                #print(ep.current_pos + ep.partition_size," / ", len(ep.edges))
                pass
        
        gt.mcmc_equilibrate(self.good_state, 
                    force_niter=10, 
                    mcmc_args={'niter': 100}, wait=2000,
                    callback=collect_edge_probs)#, verbose=(1, "A:"))
        res = [np.exp(get_avg(getIndexProbs(i))) for i in range(len(test_edg))]
        return np.transpose(np.stack([np.zeros(len(res)), res]))

        
    def reset(self):
        self.edges = None
        self.vs = None
        self.good_state = None

def save_exp_results(filename: str, results: List):
     with open(filename, "wb") as fw:
        pickle.dump({"results": results}, fw)
        fw.close()
        print("RESULTS SAVED to ", filename)

def sbm_work(data: dict, degCorrected: bool = False):
    st = time.monotonic()
    config = MySBMConfig(iterations=20, degreeCorrected=degCorrected)
    model = MySBMModel()
    score, ntime = compute_file_no_sampling_results(config, model, data)
    return score, ntime, model
    
