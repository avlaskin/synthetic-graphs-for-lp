import pickle
import time
import math
import numpy as np
import networkx as nx

from abc import ABC, abstractmethod
from datetime import datetime
from typing import List, Tuple, Dict

from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import scipy.sparse as ssp 

class GraphParams(dict):
    def __init__(self,
                 N: int,
                 cdegree:int,
                 bdegree: int,
                 ratio: float,
                 prefix: str = "",
                 ideal_auc: float = 0.0,
                 emperical_auc: float = 0.0,
                 clique: bool = False,
                 barabassi: bool = False):
        dict.__init__(self, N=N, 
                      cdegree=cdegree, 
                      bdegree=bdegree, 
                      ratio=ratio,
                      ideal_auc=ideal_auc,
                      emperical_auc=emperical_auc,
                      clique=clique, 
                      barabassi=barabassi)
        self.N = N
        self.cdegree = cdegree
        self.bdegree = bdegree
        self.ratio = ratio
        self.prefix = prefix
        self.barabassi = barabassi
        self.clique = clique
        self.ideal_auc = ideal_auc
        self.emperical_auc = emperical_auc
        
def gen_nx_graph(nodes, edges) -> nx.Graph:
    g = nx.Graph(directed=False)
    for i, n in enumerate(nodes):
        g.add_node(n)
    g.add_edges_from(edges)
    return g

def get_structure_number(node: int,
                         gp: GraphParams) -> int:
    print("Function is deprecated. Used get_structure_adjusted")
    assert False
    N = gp.N
    k = gp.cdegree
    ratio = gp.ratio
    is_clique = gp.clique
    if is_clique:
        div = k
    else:
        div = k*k
    nodes_vv = int(N*ratio / div) * div
    if node >= nodes_vv - 1:
        return -1
    return int(node // div)

def get_structure_adjusted(node1: int,
                           node2: int,
                           gp: GraphParams) -> tuple[int]:
    N = gp.N
    k = gp.cdegree
    fn = lambda x: x
    if gp.clique == False:
        fn = lambda x: x**2
    ratio = gp.ratio
    cmap = gp['cmap']
    lmap = []
    for k in cmap.keys():
        lmap.append((k, cmap[k]))
    lmap = sorted(lmap, reverse=True)
    n1 = node1
    n2 = node2
    r1 = -1 # Bridge node
    r1_size = -1
    r2 = -1 # Bridge node
    counter = 0 # Node counter
    struct_counter = 0
    for i in range(len(lmap)):
        structure_size, structure_num = lmap[i]
        for j in range(structure_num):
            counter += fn(structure_size)
            if n1 is not None and n1 < counter:
                r1 = struct_counter
                n1 = None
                r1_size = structure_size
            if n2 is not None and n2 < counter:
                r2 = struct_counter
                n2 = None
            struct_counter += 1
    return r1, r2, r1_size

def getNumberOfLinksIn2dLattice(K: int, M: int):
    """
    Returns number of edges for the KxM lattice.
    """
    g = nx.grid_2d_graph(K, M)
    return len([1 for _ in g.edges])

def get_edge_probability(edge: Tuple[int, int], gp: GraphParams, d):
    n1 = edge[0]
    n2 = edge[1]
    s1, s2, s1s = get_structure_adjusted(n1, n2, gp)
    #print("SS ", s1, " ", s2 , " ", s1s)
    N = gp.N
    k = gp.cdegree
    ratio = gp.ratio
    is_clique = gp.clique   
    if is_clique:
        div = k
    else:
        div = k*k
    nodes_vv = int(N*ratio*div / div)
    if s1 == -1 and s2 == -1:
        d['z'] += 1
        return 0
    if s1 == -1 and s2 > -1:
        # B <-> S Bdegree / Number of S nodes
        d['w'] += 1
        return gp.bdegree / nodes_vv
    if s1 > -1 and s2 == -1:
        # S <-> B
        d['w'] += 1
        div = s1s*s1s
        nodes_vv = int(N*ratio*div / div)
        return gp.bdegree / nodes_vv
    if s1 == s2 and s1 > -1:
        # for clique = 1
        # otherwise depends
        if is_clique:
            d['v'] += 1
            return 1.0
        div = s1s*s1s
        sIndex1 = n1 % div
        sIndex2 = n2 % div
        #print("S ", sIndex1, " ", sIndex2)
        # Generate lattice and see if they are connected :)
        gl = nx.grid_2d_graph(s1s, s1s)
        nodes = [n for n in gl.nodes]
        emap = [(nodes.index(e[0]), nodes.index(e[1])) for e in gl.edges]
        for e in emap:
            if e[0] == sIndex1 and e[1] == sIndex2:
                d['v'] += 1
                return 1.0
            if e[1] == sIndex1 and e[0] == sIndex2:
                d['v'] += 1
                return 1.0
    if s1 > -1 and s2 > -1 and s1 != s2:
        d['y'] += 1
    return 0.0

def calculate_perfect_auc(test_edges, test_labels, gp: GraphParams) -> float:
    """ Calculates AUC with ideal method."""
    pr = []
    pos = 0
    neg = 0
    d = {}
    d['v'] = 0
    d['y'] = 0
    d['z'] = 0
    d['w'] = 0
    d['x'] = 0
    ttt = 0
    for l, e in enumerate(test_edges):
        p = get_edge_probability(e, gp, d)
        if np.abs(p - test_labels[l]) > 0.5 and e[0] > 1799 and e[0] < 2527 and p < 0.00001:
            ttt += 1
        if p > 0.9:
            pos += 1
        if p < 0.000001:
            neg += 1
        pr.append(p)
    #print("Emperical Positives:", pos, " Negatives: ", neg)
    #print("Emperical :", d)
    res = roc_auc_score(test_labels, pr)
    print("Found anomalies ", ttt)
    return res , d
    
def computeIdealAUCFull(N: int, 
                        ratio: float, 
                        k: int, 
                        m: int, 
                        bridge_degree: int,
                        is_clique: bool) -> float:
    """
    Params:
    N - number of the verticles
    ratio - ratio between total and lattice nodes
    k - horizontal size of the lattice
    m - vertical size of the lattice
    bridge_degree - bridge degree
    """
    # Real realization of number of Structural nodes:
    if is_clique and k != m:
        raise ValueError("K != M for cliques case.")
    
    if is_clique:
        nodes_vv = int(N*ratio * k / k) 
        V = int(int(N*ratio * k / k) * (k - 1)/2)
    else:
        nodes_vv = int(N*ratio*(k*m) / (k*m)) 
        V = int(N*ratio * getNumberOfLinksIn2dLattice(k, m) / (k*m))  
    W = int((N - nodes_vv) * bridge_degree)
    Y = int(nodes_vv**2 / 2) - V
    X = int((N - nodes_vv)*nodes_vv) - W
    Z = int((N - nodes_vv)**2 / 2)
    T = int(N*N/2)
    idealAUC =  (1 - 0.5 * (1.0 - V / (V + W)) * X / (X + Y + Z) )
    print("Ideal N = %d Structural Nodes: %d" % (N, nodes_vv))
    return idealAUC, T, V, W, X, Y, Z

def computeIdealAUC(N: int, ratio: float, k: int, bridge_degree: int, is_clique: bool) -> float:
    return computeIdealAUCFull(N=N, ratio=ratio, k=k, m=k, bridge_degree=bridge_degree, is_clique=is_clique)

def computeIdealiSBM(N: int, ratio: float, k: int, 
                     bridge_degree: int, is_clique: bool,
                     clique_prob: float = 1.0) -> float:
    if is_clique:
        nodes_vv = int(N*ratio * k / k) 
        if clique_prob < 1.0:
            V = int(int(clique_prob*N*ratio * k / k) * (k - 1)/2)
        else:    
            V = int(int(N*ratio * k / k) * (k - 1)/2)
        B = int(int(N*ratio * k / k) * (k - 1)/2)
        Q = nodes_vv*(nodes_vv - k + 1) / 2
    else:
        nodes_vv = int(N*ratio*(k*k) / (k*k)) 
        V = int(N*ratio * getNumberOfLinksIn2dLattice(k, k) / (k*k))
        B = int(N*ratio*k*k*(k*k - 1)/(2*k*k)) # int(N*ratio*k*k*(k*k - 1)/(2*k*k))
        Q = nodes_vv*(nodes_vv - k*k + 1) / 2
    #assert abs(nodes_vv*nodes_vv/2 - B - Q) < 1000, "Q and B and N should follow rules: %f %f %f" % (abs(nodes_vv*nodes_vv/2 - B - Q), Q, B)
    if abs(nodes_vv*nodes_vv/2 - B - Q) > 100:
        print("DEVIATION : ", abs(nodes_vv*nodes_vv/2 - B - Q))
    print("Lattice links ",  getNumberOfLinksIn2dLattice(k, k) , nodes_vv*nodes_vv/2 - B, Q)
    
    Z = int((N - nodes_vv)**2 / 2)
    W = (N - nodes_vv)*bridge_degree
    G = int(nodes_vv*N*(1 - ratio))
    SS = B - V
    BS = G - W
    TS = Q + Z
    AS = BS + SS + TS
    Pp = bridge_degree / int(N * ratio)
    Pq = 2 * getNumberOfLinksIn2dLattice(k, k) / (k*k*(k*k - 1))
    if Pq > Pp:
        M = V / (V + W)
        return 1 - (1 - M)*SS/AS - 0.5*(1 - M)*BS/AS - 0.5*M*SS/AS
    else:
        M = W / (V + W)
        return 1 - (1 - M)*BS/AS - 0.5*(1 - M)*SS/AS - 0.5*M*BS/AS
    #OLD CODE:
    #return 1 - 0.5 * (( 2*SS + BS )/(SS + BS + TS) - (V/(V+W))*((SS + BS) / (SS + BS +TS)))


def convert_to_graph(adj_matrix: np.array, directed: bool = False):
    '''Converts adj matrix to the nx.Graph.'''
    nodes = [x for x in range(0, len(adj_matrix))]
    edges = []
    for i in range(len(adj_matrix)):
        for j in range(len(adj_matrix)):
            if adj_matrix[i,j] > 0.0:
                #print(i, "->", j)
                edges.append((i, j))
    G = nx.DiGraph() if directed else nx.Graph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)
    return G

def get_adj(nodes: List, edges: List, directed=False) -> np.array:
    N = len(nodes)
    adj = np.zeros((N, N), dtype=np.float32)
    for e in edges:
        adj[e[0]][e[1]] = 1.0
        if not directed:
            adj[e[1]][e[0]] = 1.0
    return adj

def generate_sparse_adj(N:int, edges: list[tuple[int, int]], exclude_edges: list[tuple[int, int]]) -> ssp.csc_matrix:
    """Function generates full adjucency sparse matrix."""
    edge_1 = [e[0] for e in edges]
    edge_2 = [e[1] for e in edges]
    ne1 = [e[0] for e in exclude_edges]
    ne2 = [e[1] for e in exclude_edges]
    net = ssp.csc_matrix(                                                              
       (np.ones(len(edges)), (edge_1, edge_2)),                                        
        shape=(N, N)                                                                        
    )                                                                                                                             
    net[edge_2, edge_1] = 1  # add symmetric edges 
    net[np.arange(N), np.arange(N)] = 0  # remove self-loops
    net[ne1, ne2] = 0  # mask excluded links                                                                      
    net[ne2, ne1] = 0  # mask excluded links 
    return net

def find_edges_with(node: int, edges: List):
    res = []
    for e in edges:
        if e[0] == node or e[1] == node:
            res.append(e)
    return res

def extend_to_symmetric(edges: List[Tuple[int]]):
    ls = set(edges)
    s = set(edges)
    for e in ls:
        s.add((e[1], e[0]))
    return list(s)

def sample_positive_edges(nod: List, 
                          edg: List, 
                          labels: List,
                          percent: float = 0.2) -> List:
    '''Simplest version of sampling from cliques. '''
    ## TODO: Add more complex version where we sample a few vertices 
    ##       in the graph and 
    labt = [1 if l > 0 else 0 for l in labels]
    num_test = int(math.floor(percent * sum(labt)))
    r = np.random.choice(list(range(sum(labt))), size=num_test, replace=False)
    result = []
    for n in r:
        edges_to_skip = labels[n] - 2
        all_node_edges = find_edges_with(n, edg)
        for se in all_node_edges[:edges_to_skip]:
            result.append(se)
    return result

def sample_better_positive_edges(nod: List,
                                 edg: List,
                                 labels: List) -> List:
    result = []
    n = 0
    while n < len(nod):
        l = labels[n] 
        if l == 0:
            break
        all_node_edges = find_edges_with(n, edg)
        delta = 0
        if l == 3:
            # here we take away just one edge
            r = 1            
        elif l > 3:
            all_node_edges += find_edges_with(n+2, edg)
            r = l - 1
        idxs = np.random.choice(list(range(0, len(all_node_edges))), size=r, replace=False)
        for i in idxs:
            result.append(all_node_edges[i])
        n += l
    return result

def sample_even_better_positive_edges(nod: List,
                                 edg: List,
                                 labels: List,
                                 edge_labels: List) -> List:
    result = []
    clique_edges = edg[:len([1 for i in range(len(edg)) if edge_labels[i] > 0])]
    n = 0
    while n < len(nod):
        l = labels[n] 
        if l == 0:
            break
        all_node_edges = find_edges_with(n, clique_edges)
        delta = 0
        if l == 3:
            # here we take away just one edge
            r = 1            
        elif l > 3 and l <= 6:
            all_node_edges += find_edges_with(n+1, clique_edges)
            r = l - 1
        elif l > 6:
            all_node_edges += find_edges_with(n+1, clique_edges)
            all_node_edges += find_edges_with(n+2, clique_edges)
            r = int(l*1.2)
        idxs = np.random.choice(list(range(0, len(all_node_edges))), size=r, replace=False)
        for i in idxs:
            result.append(all_node_edges[i])
        n += l
    return result

def sample_random_positive_edges(nod: List,
                                 edg: List,
                                 edge_labels: List,
                                 percent: float = 10) -> List:
    size = int((len(edg) * percent) / 100)
    idxs = np.random.choice(list(range(0, len(edg))), size=size, replace=False)
    res = []
    for i in idxs:
        res.append(edg[i])
    return res

def sample_random_positive_edges_fair(nod: List,
                                     edg: List,
                                     edge_labels: List,
                                     how_many: int) -> List:
    size = how_many
    idxs = np.random.choice(list(range(0, len(edg))), size=size, replace=False)
    res = []
    for i in idxs:
        res.append(edg[i])
    return res
    

def sample_negative_edges(nod: List, 
                          edg: List,
                          num_samples: int) -> List:
    mn = max(nod)
    nset = set(nod)
    eset = set(edg)
    res = set()
    while len(res) < num_samples:
        i = np.random.randint(0, mn)
        j = np.random.randint(0, mn)
        if i != j and i < j and i in nset and j in nset and (i, j) not in eset and (j, i) not in eset and (i, j) not in res and (j, i) not in res:
            res.add((i, j))
    return list(res)

def operator_hadamard(u, v):
    return u * v

def operator_l1(u, v):
    return np.abs(u - v)

def operator_l2(u, v):
    return (u - v) ** 2

def operator_avg(u, v):
    return (u + v) / 2.0

def operator_dot(u, v):
    return np.dot(u, v)

def operator_argcos(u, v):
    n1 = np.dot(u, u)
    n2 = np.dot(v, v)
    return np.dot(u, v) / (n1 * n2) 

class Configuration(ABC):
    '''Model config abstract class.''' 
    @abstractmethod
    def isDirected(self):
        pass

class AbsModel(ABC):
    ''' Model abstract class.'''
    @abstractmethod
    def train(self, config, 
              train_labels: List, 
              train_edg: List,
              real_edges: List,
              real_nodes: List):
        pass
    
    @abstractmethod
    def predict(self, test_features: List):
        pass
    
    def compute_auc(self, predicted: List, test_labels: List):
        assert len(predicted.shape) > 1 and len(test_labels) == predicted.shape[0]
        #pr = [0.0 if p[0] > p[1] else 1.0 for p in predicted]
        pr = [p[1] for p in predicted]
        res = roc_auc_score(test_labels, pr)
        return res
    
    def saveModel(self, filename: str):
        pass
    
    @abstractmethod
    def reset(self):
        pass

class MySimpleConfig(Configuration):
    ''' Simplest config for heuristics models.'''
    def __init__(self):
        self.directed = False
        
    #override
    def isDirected(self):
        return self.directed


def find_neighbours(n: int, 
                    edges: List[Tuple]):
    res = set()
    for e in edges:
        if e[0] == n:
            res.add(e[1])
        if e[1] == n:
            res.add(e[0])
    return res
        
def jaccard_coef_score(e: Tuple, edges: List) -> float:
    l1 = find_neighbours(e[0], edges)
    l2 = find_neighbours(e[1], edges)
    intersect = l1.intersection(l2)
    union = l1.union(l2)
    if len(union) > 0:
        return len(intersect) / len(union)
    return 0

def adamic_score(e: Tuple, edges: List) -> float:
    l1 = find_neighbours(e[0], edges)
    l2 = find_neighbours(e[1], edges)
    intersect = l1.intersection(l2)
    summ = 0.0
    for x in intersect:
        lx = find_neighbours(x, edges)
        if len(lx) > 1:
            summ += 1.0 / math.log(len(lx))
        elif len(lx) == 0:
            summ += 3.0
        elif len(lx) == 1:
            summ += 2.0 
    return summ
        

class MySimpleScoreModel(AbsModel):
    '''This model is for simple heuristic link predictions.'''
    def __init__(self, func=jaccard_coef_score):
        self.edges = None
        self.func = func
    
    def train(self, 
              config,  
              train_labels: List, 
              train_edg: List,
              real_nodes: List,
              real_edges: List):
        self.edges = real_edges
    
    def predict(self, test_edg):
        res = []
        for i in range(len(test_edg)):
            s = self.func(test_edg[i], self.edges)
            if s > 0:
                res.append([0.0, s])
            else:
                res.append([1.0, 0.0])
        return np.array(res, dtype=np.float32)
    
    def reset(self):
        self.edges = None

def get_graph_data(fname) -> Dict:
    data = None
    with open(fname, "rb") as fr:
        data = pickle.load(fr)
        fr.close()
    return data


def compute_results(conf: Configuration, 
                    model: AbsModel,
                    graph_sufixes: List[int]):
    '''Trains func returns a set of results and timings.'''
    auc_scores = []
    time_took = []
    for n in graph_sufixes:
        ## READING DATA
        st = time.monotonic()
        d400 = get_graph_data("graph_ee_%d.pkl" % n)
        nod, edg, lab = d400['nodes'], d400['edges'], d400['labels']
        
        ## SAMPLING AND EXCLUDE FEATURE LEAKAGE
        pos = sample_better_positive_edges(nod, edg, lab)
        edg = list(set(edg) - set(pos))
        
        not_edg = sample_negative_edges(nod, edg, len(edg) + len(pos))
        pos, neg = pos, not_edg[len(edg):]
        
        print("Nodes: %d Edges: %d Positive: %d Negative: %d" % (len(nod), len(edg), len(pos), len(neg)))
        train_edg = not_edg[:len(edg)] + edg
        train_labels = np.hstack([np.zeros(len(edg)), np.ones(len(edg))])
        test_labels = np.hstack([np.zeros(len(neg)), np.ones(len(pos))])
        test_edg = neg + pos
        
        not_edg_set = set(not_edg[:len(edg)])
        edg_set = set(edg) 
        
        assert not not_edg_set.intersection(set(neg)) 
        assert not edg_set.intersection(set(pos))
        st2 = time.monotonic()
        model.train(conf, train_labels, train_edg, nod, edg)      
        predicted = model.predict(test_edg)
        auc_score = model.compute_auc(predicted, test_labels)
        st3 = time.monotonic()
        nod, edg, graph, A, emb = None, None, None, None, None
        print()
        auc_scores.append(auc_score)
        time_took.append(st3 - st2)
        print("AUC Score : ", auc_score, "\n Stage 1 took: ", st2 - st, " Stage 2: ", st3 - st2)
        model.reset()
        print("Scores: ", auc_scores, " Times: ", time_took)
    return auc_scores, time_took

def compute_no_sampling_results(conf: Configuration, 
                    model: AbsModel,
                    graph_sufixes: List[int],
                    prefix: str = "gg",
                    file_index: str = None,
                    folder: str = "."):
    '''Trains func returns a set of results and timings.'''
    auc_scores = []
    time_took = []
    for n in graph_sufixes:
        ## READING DATA
        st = time.monotonic()
        print("Started training: ", datetime.now())
        if file_index:
            fname = "%s/graph_%s_%d_%s.pkl" % (folder, prefix, n, file_index)
        else:
            fname = "%s/graph_%s_%d.pkl" % (folder, prefix, n)
        print("Opening data file: ", fname)
        data = get_graph_data(fname)
        nod, edg = data['nodes'], data['edges']
        
        ## SAMPLING AND EXCLUDE FEATURE LEAKAGE
        #pos = data['positives']
        pos = list(set(data['positives']))
        edg = list(set(edg) - set(pos))
        
        not_edg = data['negatives']
        pos, neg = pos, not_edg[len(edg):len(edg) + len(pos)]
        if len(neg) > len(pos):
            print("ATTENTION: Overlapping Train and Test datasets!!!")
            neg = neg[len(neg) - len(pos):]
        
        print("Nodes: %d Edges: %d Positive: %d Negative: %d" % (len(nod), len(edg), len(pos), len(neg)))
        train_edg = not_edg[:len(edg)] + edg
        train_labels = np.hstack([np.zeros(len(edg)), np.ones(len(edg))])
        
        test_labels = np.hstack([np.zeros(len(neg)), np.ones(len(pos))])
        test_edg = neg + pos
        
        print("Total test edges: ", len(test_edg), " Positive: ", len(pos))
        
        not_edg_set = set(not_edg[:len(edg)])
        edg_set = set(edg) 
        print("AV Train: %d %d Tets: %d %d" % (len(train_edg), len(train_labels), len(test_edg), len(test_labels) ))
        assert not not_edg_set.intersection(set(neg)) 
        assert not edg_set.intersection(set(pos))
        assert not set(train_edg).intersection(set(test_edg))

        st2 = time.monotonic()
        model.train(conf, train_labels, train_edg, nod, edg)    
        print("Started predicting: ", datetime.now())
        predicted = model.predict(test_edg)
        #print(test_labels)
        #print(predicted)
        auc_score = model.compute_auc(predicted, test_labels)
        st3 = time.monotonic()
        nod, edg, graph, A, emb = None, None, None, None, None
        print()
        auc_scores.append(auc_score)
        time_took.append(st3 - st2)
        print("AUC Score : ", auc_score, "\n Stage 1 took: ", st2 - st, " Stage 2: ", st3 - st2)
        model.reset()
        print("Scores: ", auc_scores, " Times: ", time_took)
    return auc_scores, time_took

def compute_file_no_sampling_results(conf: Configuration, 
                                     model: AbsModel,
                                     data: dict) -> tuple[float, float]:
    '''Trains func returns a set of results and timings.'''
    if True:
        ## READING DATA
        st = time.monotonic()
        nod, edg = data['nodes'], data['edges']        
        ## SAMPLING AND EXCLUDE FEATURE LEAKAGE
        #pos = data['positives']
        pos = list(set(data['positives']))
        edg = list(set(edg) - set(pos))
        
        not_edg = data['negatives']
        pos, neg = pos, not_edg[len(edg):len(edg) + len(pos)]
        if len(neg) > len(pos):
            print("ATTENTION: Overlapping Train and Test datasets!!!")
            neg = neg[len(neg) - len(pos):]
        
        print("Nodes: %d Edges: %d Positive: %d Negative: %d" % (len(nod), len(edg), len(pos), len(neg)))
        train_edg = not_edg[:len(edg)] + edg
        train_labels = np.hstack([np.zeros(len(edg)), np.ones(len(edg))])        
        test_labels = np.hstack([np.zeros(len(neg)), np.ones(len(pos))])
        test_edg = neg + pos
        
        print("Total test edges: ", len(test_edg), " Positive: ", len(pos))
        
        not_edg_set = set(not_edg[:len(edg)])
        edg_set = set(edg) 
        print("AV Train: %d %d Tets: %d %d" % (len(train_edg), len(train_labels), len(test_edg), len(test_labels) ))
        assert not not_edg_set.intersection(set(neg)) 
        assert not edg_set.intersection(set(pos))
        assert not set(train_edg).intersection(set(test_edg))

        st2 = time.monotonic()
        if conf.__dict__.get('multi_sample', 1) > 1:
            print('Multisample is ON with : ',conf.__dict__.get('multi_sample', 1)) 
            preds = []
            for _ in range(conf.__dict__.get('multi_sample', 1)):
                model.train(conf, train_labels, train_edg, nod, edg)    
                predicted = model.predict(test_edg)
                preds.append(predicted)
            predicted = np.average(preds, axis=0)
            auc_score = model.compute_auc(predicted, test_labels)
            st3 = time.monotonic()
            nod, edg, graph, A, emb = None, None, None, None, None
            time_took = st3 - st
        else:
            print('Multisample is OFF')
            model.train(conf, train_labels, train_edg, nod, edg)    
            predicted = model.predict(test_edg)
            assert len(predicted) == len(test_labels)
            auc_score = model.compute_auc(predicted, test_labels)
            st3 = time.monotonic()
            nod, edg, graph, A, emb = None, None, None, None, None
            time_took = st3 - st
        return auc_score, time_took
    return None, None
