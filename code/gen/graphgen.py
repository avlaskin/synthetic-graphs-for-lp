import math
import numpy as np
import networkx as nx
import random
import time

from collections import deque, defaultdict
from typing import List, Tuple
from graphparams import GraphParams, save_graph

from experiments import sample_random_positive_edges
from experiments import calculate_perfect_auc

def get_adj(nodes: List, edges: List, directed=False) -> np.array:
    N = len(nodes)
    adj = np.zeros((N, N), dtype=np.float32)
    for e in edges:
        adj[e[0]][e[1]] = 1.0
        if not directed:
            adj[e[1]][e[0]] = 1.0
    return adj

def make_star(num_per=20, 
              c = (.0, .0), 
              linked=False,
              start_index=0,
              R=1.0,
              clique=False):
    angle = 0
    step = 2.0 * np.pi / num_per
    edges = []
    nodes = [start_index]
    nodes_emb = [c]
    j = start_index + 1
    for i in range(num_per):
        angle += step
        x = c[0] + R*np.sin(angle)
        y = c[1] + R*np.cos(angle)
        nodes_emb.append((x, y))
        if not clique:
            edges.append((start_index, j))
        nodes.append(j)
        if linked and j > (start_index + 1):
            edges.append((j-1, j))
        j += 1
    if linked and not clique:
        edges.append((start_index + 1, j-1))
    if clique:
        edges = []
        nodes = []
        nodes_emb = []
        for i in range(num_per):
            angle += step
            x = c[0] + R*np.sin(angle)
            y = c[1] + R*np.cos(angle)
            nodes_emb.append((x, y))
            nodes.append(i+start_index)
            for j in range(i+1, num_per):
                edges.append((i+start_index, j+start_index))
    return nodes, edges, nodes_emb

def make_m_stars(num_of_stars:int, num_per_row:int, num_per:int, linked = False, clique=False):
    assert num_of_stars % num_per_row == 0, " Number of stars must be divisible by number of rows."
    divider = num_of_stars // num_per_row
    nodes = []
    edges = []
    node_emb = []
    node_index = []
    star_index = 0
    start_index = 0
    for i in range(divider):
        for j in range(num_per_row):
            Cx = j * 1.0
            Cy = i * 1.0
            n, e, nemb = make_star(num_per=num_per, c=(Cx, Cy), linked=linked, start_index=start_index, R=0.45, clique=clique)
            nodes += n
            edges += e
            node_index += [star_index]*len(n)
            node_emb += nemb
            start_index = len(nodes)
            star_index += 1
    return nodes, edges, node_emb, node_index


def remove_solitary(gr):
    solitary = [i for i in range(len(gr.nodes())) if gr.degree[i] == 0]
    gr.remove_nodes_from(solitary)

def add_random(nodes: List, 
               edges: List, 
               embedding: List = [],
               center=(0.0, 0,0),
               R = 1.05, 
               ratio=1.0, 
               prob=0.1,
              max_range=2.5):
    N2 = int(len(nodes)*ratio)
    N1 = len(nodes)
    for i in range(N2):
        nodes.append(N1+i)
        while len(embedding) > 0:
            x = -1.0*max_range/2 + max_range*np.random.random() + center[0]
            y =  -1.0*max_range/2 + max_range*np.random.random() + center[1]
            if math.sqrt((x - center[0])*(x - center[0]) + (y - center[1])*(y - center[1])) > R:
                embedding.append((x, y))
                break
    for i in range(len(nodes)):
        for j in range(i+1, len(nodes)):
            r = np.random.random()
            if r < prob:
                edges.append((i, j))
    return nodes, edges, embedding

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

def make_graph(nodes: List[int], 
               edges: List[Tuple[int]],
               directed: bool = False):
    G = nx.DiGraph() if directed else nx.Graph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)
    return G


def is_connected_nx(graph: nx.Graph) -> bool:
    ccs = [len(c) for c in sorted(nx.connected_components(graph), key=len, reverse=True)]
    return len(ccs) == 1

def is_connected(adj_matrix: np.array):
    '''Checks if Graph is connected. '''
    stack = deque()
    n = 0
    visited = set()
    stack.append(n)
    while stack:
        c = stack.popleft()
        visited.add(c)
        ful_cons = adj_matrix[c,:]
        connections = list(set([x for x in range(len(ful_cons)) if ful_cons[x] > 0]))
        stack += list(set(connections) - visited)
    all_set = set(list(range(len(adj_matrix))))
    diff_set = all_set - visited
    return len(diff_set) == 0

def create_graph(directed: bool, 
                 tau: float=0.5, 
                 c_min: int=10, 
                 c_max=18, 
                 N=100, 
                 mu=0.7, 
                 degree=2):
    powerd = tfp.distributions.Exponential(tau)
    if N < 1000:
        binsd = 1000
    else:
        binsd = N*10
    k = N
    s = 0
    while s < N: 
        ss = powerd.sample(k)
        b = np.histogram(ss.numpy(), bins=binsd)
        bins = b[0][c_min:c_max]
        k = int(k * 1.1)
        s = sum(bins)
    m = sum(bins)
    adj = np.zeros((m,m))
    p_in_community = mu
    p_outside = 1. - mu
    s = 0
    e = 0
    print(bins)
    edges = []
    for c in bins:
        e = s + c
        print(s, e)
        outside = list(range(0, s)) + list(range(e, m))
        inside = list(range(s, e))
        for node in range(c):
            for _ in range(degree):
                if c == 0:
                    pass
                r = np.random.random()
                cur = s + node
                comp = ""
                linked = None
                if r > mu:
                    ## new edge ouside of the community
                    comp = "Outside"
                    linked = np.random.choice(outside, 1)[0]
                else:
                    comp = "Inside"
                    ##  new edge inside the community
                    linked = np.random.choice(inside, 1)[0]
                if linked != cur:
                    print(comp + " community "+str(cur) + " " + str(linked))
                    edges.append((cur, linked))
                    adj[cur, linked] = 1.0
                    if not directed:
                        adj[linked, cur] = 1.0
        s += c
        # DBG
        #if c == bins[1]:
        #    break
    return edges, adj

def sample_natural_with_limits(distr, lmin: int, lmax: int):
    t = distr.sample(1).numpy()[0]
    while math.floor(t) < lmin or math.floor(t) > lmax:
        t = distr.sample(1).numpy()[0]
    return math.floor(t)

def create_graph_with_degree(directed: bool, 
                 tau: float = 0.5, 
                 tau2: float = 0.2,
                 c_min: int = 10, 
                 c_max: int = 18,
                 d_min: int = 1, 
                 d_max: int = 12,
                 N:     int = 100, 
                 mu:  float = 0.7):
    community_distr = tfp.distributions.Exponential(tau)
    degree_distr = tfp.distributions.Exponential(tau2)
    if N < 1000:
        binsd = 1000
    else:
        binsd = N*10
    k = N
    s = 0
    while s < N: 
        ss = community_distr.sample(k)
        b = np.histogram(ss.numpy(), bins=binsd)
        bins = b[0][c_min:c_max]
        k = int(k * 1.1)
        s = sum(bins)
    m = sum(bins)
    adj = np.zeros((m,m))
    p_in_community = mu
    p_outside = 1. - mu
    s = 0
    e = 0
    print(bins)
    edges = []
    for c in bins:
        e = s + c
        print(s, e)
        outside = list(range(0, s)) + list(range(e, m))
        inside = list(range(s, e))
        for node in range(c):
            degree = sample_natural_with_limits(degree_distr, d_min, d_max)
            for _ in range(degree):
                if c == 0:
                    pass
                r = np.random.random()
                cur = s + node
                linked = None
                if r > mu:
                    linked = np.random.choice(outside, 1)[0]
                else:
                    linked = np.random.choice(inside, 1)[0]
                if linked != cur:
                    edges.append((cur, linked))
                    adj[cur, linked] = 1.0
                    if not directed:
                        adj[linked, cur] = 1.0
        s += c
    return edges, adj

def generate_graph_with_degree(directed: bool, 
                 tau: float = 0.5, 
                 tau2: float = 0.2,
                 c_min: int = 10, 
                 c_max: int = 18,
                 d_min: int = 1, 
                 d_max: int = 12,
                 N:     int = 100, 
                 mu:  float = 0.7):
    community_distr = tfp.distributions.Exponential(tau)
    degree_distr = tfp.distributions.Exponential(tau2)
    # Sampling community structure
    m = 0
    bins = []
    degrees = []
    while m < N:
        c = sample_natural_with_limits(community_distr, c_min, c_max)
        bins.append(c)
        m += c
    bins.sort(reverse=True)
    m = sum(bins)
    adj = np.zeros((m,m))
    p_in_community = mu
    p_outside = 1. - mu
    s = 0
    e = 0
    edges = []
    for c in bins:
        e = s + c
        outside = list(range(0, s)) + list(range(e, m))
        inside = list(range(s, e))
        for node in range(c):
            degree = sample_natural_with_limits(degree_distr, d_min, d_max)
            degrees.append(degree)
            for _ in range(degree):
                if c == 0:
                    pass
                r = np.random.random()
                cur = s + node
                linked = None
                if r > mu:
                    linked = np.random.choice(outside, 1)[0]
                else:
                    linked = np.random.choice(inside, 1)[0]
                if linked != cur:
                    edges.append((cur, linked))
                    adj[cur, linked] = 1.0
                    if not directed:
                        adj[linked, cur] = 1.0
        s += c
    return edges, adj, degrees, bins

def find_edges_with(node: int, edges: List):
    res = []
    for e in edges:
        if e[0] == node or e[1] == node:
            res.append(e)
    return res

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

def sample_negative_edges(nod: List, 
                          edg: List,
                          nodes_num: int) -> List:
    mn = max(nod)
    nset = set(nod)
    eset = set(edg)
    res = set()
    while len(res) < nodes_num:
        i = np.random.randint(0, mn)
        j = np.random.randint(0, mn)
        if i != j and i in nset and j in nset and (i, j) not in eset and (j, i) not in eset and (i, j) not in res and (j, i) not in res:
            res.add((i, j))
    return list(res)

def remove_solitary(nod: List, 
                    edg: List,
                    labels: List) -> Tuple[List, List, List]:
    not_sol = set([e[0] for e in edg] + [e[1] for e in edg])
    to_remove = []
    
    new_edges = [(0, 0)]*len(edg)
    
    for n in nod:
        if n not in not_sol:
            to_remove.append(n)
    if not to_remove:
        return nod, edg
    to_remove_set = set(to_remove)
    new_labels = [0]*(len(nod) - len(to_remove))
    new_nodes = [0]*(len(nod) - len(to_remove))
    c = 0
    for j, n in enumerate(nod):
        if n in to_remove_set:
            continue
        new_labels[c] = labels[j] 
        new_nodes[c] = n
        c += 1
    new_aligned = list(range(len(new_nodes)))
    # Now we have a mapping between old values 'new_nodes' and new ones 'new_aligned'
    # and we need to map the edges to the new ones
    for j, e in enumerate(edg):
        id0 = new_nodes.index(e[0])
        id1 = new_nodes.index(e[1])
        new_edges[j] = (new_aligned[id0], new_aligned[id1])
    
    return new_aligned, new_edges, new_labels

## Not needed on top

def add_clique(start_index:int, 
               K:int, 
               add_nodes:bool,
               nodes: list[int], 
               edges: list[tuple],
               ratio: float = 1.0) -> tuple[int, int]:
    """Creates a clique in the graph."""
    n,e = 0,0
    if add_nodes:
        for i in range(K):
            n += 1
            nodes.append(start_index + i)
    else:
        n = K
    interim_edges = []
    
    for i in range(K):
        for j in range(i+1, K):
            interim_edges.append((i+start_index, j+start_index))
            e += 1
    if e != (n*(n-1) // 2):
        print("Invalid number of edges: ", e, (n*(n-1) // 2))
    if abs(ratio - 1.0) < 0.001:
        edges += interim_edges
        return n, e   
    total = len(interim_edges)
    edges_indexes = np.random.choice(list(range(len(interim_edges))), 
                                     int(ratio*total), 
                                     replace=False)
    addition = [tuple(x) for x in list(np.array(interim_edges, dtype=np.int64)[edges_indexes])]
    edges += addition
    e = len(addition)
    return n, e
    
def add_bridge(start_index:int, 
            num_edges: int, 
            nodes: list[int], 
            edges: list[Tuple],
            logits: list[float] = None):
    if not logits:
        logits = [1.0 for _ in nodes] # Uniform logits
    pr = tfp.distributions.Categorical(logits=logits)
    ed = pr.sample(num_edges).numpy()
    for j in range(num_edges):
        if start_index != ed[j]:
            edges.append((start_index, ed[j]))

def generate_bridges(nodes: list,
                     edges: list,
                     bridge_start: int,
                     bridge_end: int, 
                     bridge_degree: float):
    # 1. Generate all bridge nodes first 
    all_bridge_nodes = list(range(bridge_start, bridge_end))
    # 2. Create two lists (Structures) & (Bridges)
    all_structure_nodes = list(range(bridge_start))
    # 3. Create a number of links that we would like to get. Using bridge average degree.
    num_links = int( (bridge_end - bridge_start) * bridge_degree)
    # 4. Generate random connections between structure nodes and bridge nodes.
    sedges = set(edges)
    #print("Generating %d S-B edges" % num_links)
    for _ in range(num_links):
        going = True
        while going:
            b = random.randrange(bridge_start, bridge_end) # bridge_Start <= s <= bridge_end-1
            s = random.randrange(0, bridge_start) # 0 <= s <= bridge_start-1
            if (s, b) not in sedges:
                edges.append((s, b))
                sedges = set(edges)
                going = False
                break
    return edges

def generate_fixed_clique_graph(N: int, 
                                clique_degree: int,
                                bridge_degree: int, 
                                cc: float,
                                new_bridge: bool=True,
                                fill: bool=True,
                                structure_ratio: float = 1.0):
    '''
    cc - proportion of clique nodes. Cb= 1 - cc. c belongs to (0; 1).
    K -  number of clique nodes K = N * cc
    M -  number of bridge nodes M = N *( 1 - cc)
    bridge nodes only connect to clique nodes. 
    Returns:
       - (nodes, edges, labels) tuple
    '''
    K = int(N * cc)
    number_of_cliques = int(K / clique_degree)
    
    nodes = list(range(N))
    edges = []
    edge_labels = []
    cur = 0
    cmap = { clique_degree: number_of_cliques}
    clique_sizes = [clique_degree]
    if fill:
        M = clique_degree**2
        leftOver = K - M*number_of_cliques
        for l in range(clique_degree-1,1,-1):
            m = l*l
            if leftOver > m:
                if l in cmap:
                    val = cmap[l] + 1
                else:
                    val = 1
                leftOver -= l*l
                cmap[l] = val
    for j in cmap.keys():
        for i in range(cmap[j]):
            n, e = add_clique(K=j, start_index=cur, 
                              add_nodes=False, nodes=nodes, 
                              edges=edges, ratio=structure_ratio)
            cur += j
    print("Number of cliques: ", cmap)
    st_links = len(edges)
    edge_labels += [1]*st_links
    logits = [1.0 for _ in range(cur)]
    ## Secondly, Add bridge nodes
    label = cur
    if new_bridge:
        edges = generate_bridges(nodes, edges, cur, N,  bridge_degree)
    else:
        while cur < N:
            add_bridge(cur, bridge_degree, nodes=nodes, edges=edges, logits=logits)
            cur += 1
    labels = [1]*label + [0]*(N - label)
    edge_labels += [0]*(len(edges) - st_links)
    return nodes, edges, labels, edge_labels, cmap



def add_lattice(K: int, start_index: int, nodes: list[int], edges: list, diags: int=0):
    g = nx.grid_2d_graph(K, K)
    n = [x for x in g.nodes]
    ne = []
    if (0x01 & diags) > 0:
        nodes = set([x[0] for x in g.edges] + [x[1] for x in g.edges])
        for l in range(K-1):
            for j in range(K-1):
                ne.append((j+K*l+start_index, K+j+1+K*l+start_index))
                if (0x02 & diags) > 0:
                    ne.append((j+1+K*l, K+j+K*l))
    for e in g.edges:
        ne.append((n.index(e[0]) + start_index, n.index(e[1]) + start_index))
    edges += ne
    return start_index + len(n)

def generate_fixed_lattice_graph(N: int, 
                                 clique_degree: int,
                                 bridge_degree: int, 
                                 cc: float,
                                 bashrabassi: bool,
                                 new_bridge: bool=True,
                                 forcelink: bool = False,
                                 diags: int = 0,
                                 align_bridges: bool = False,
                                 fill=True):
    '''
    cc - proportion of clique nodes. Cb= 1 - cc. c belongs to (0; 1).
    K -  number of clique nodes K = N * cc
    M -  number of bridge nodes M = N *( 1 - cc)
    bridge nodes only connect to clique nodes. 
    bashrabassi - preferential connections. Not used in current graphs.
    new_bridge - when we add bridges on average.
    forcelink - creates a special type of graphs that have more structural nodes than bridges.
    diags: - adds diagonals to the graphs. 0, 1, 2 or 3.
    fill - top up with smaller graphs.
    align_bridges - special mode that generates full structures first and adds
      same number of bridges afterwards.
    Returns:
       - (nodes, edges, labels) tuple
    '''
    K = int(N * cc)
    number_of_cliques = int(K / (clique_degree**2))
    nodes = list(range(N))
    edges = []
    cur = 0
    first_cmap = None # Only used in align_bridges
    cmap = { clique_degree: number_of_cliques}
    edge_labels = []
    labels = []
    first_cmap = number_of_cliques
    if fill:
        M = clique_degree**2
        leftOver = K - M*number_of_cliques
        for l in range(clique_degree-1,1,-1):
            m = l*l
            if leftOver > m:
                if l in cmap:
                    val = cmap[l] + 1
                else:
                    val = 1
                leftOver -= l*l
                cmap[l] = val

    print("Number of lattices: ", number_of_cliques, cmap)
    if forcelink:
        firstnode = []
    for j in cmap.keys():
        for i in range(cmap[j]):
            prev_cur = cur
            cur = add_lattice(K=j, start_index=cur, nodes=nodes, edges=edges, diags=diags)
            labels += [1]*(cur - prev_cur)
            if forcelink:
                firstnode.append(cur)
    st_links = len(edges)
    edge_labels += [1]*st_links
    # print('Labels: ', len(labels))
    if forcelink:
        #1. Add one bridge node
        nodes.append(cur)
        #2. Add all links to this pridge
        edges.append((cur, 0))
        for cc in firstnode:
            if cc < N:
                edges.append((cur, cc))
        edge_labels += [0]*(len(edges) - st_links)
        labels = [1]*N + [0]
        return nodes, edges, labels, edge_labels, cmap
    
    if align_bridges:
        N = cur + first_cmap
        nodes = list(range(cur + first_cmap))
        edges = generate_bridges(nodes, edges, cur, N,  bridge_degree)
        labels += [0]*first_cmap
        edge_labels += [0]*(len(edges) - st_links)
        print('Labels: ', len(labels), ' not null ', sum(labels))
        print('Edge Labels: ', len(edge_labels))
        return nodes, edges, labels, edge_labels, cmap
    if bashrabassi:
        nodes = nodes[:cur]
        labels = [1]*len(nodes) + [0]*(N - len(nodes))
        DIRECTED = False
        adj = get_adj(nodes, edges, DIRECTED)
        graph = convert_to_graph(adj, DIRECTED)
        graph = nx.generators.barabasi_albert_graph(N, bridge_degree, None, graph)
        nodes = [x for x in graph.nodes]
        edges = [e for e in graph.edges]
        edge_labels += [0]*(len(edges) - st_links)
    else:
        logits = [1.0 for _ in range(cur)]
        ## Secondly, Add bridge nodes
        label = cur
        if new_bridge:
            #print("Generating bridges: ", N - len(labels), cur)
            edges = generate_bridges(nodes, edges, cur, N,  bridge_degree)
            labels += [0]*(N-label)
            #print('Labels: ', len(labels))
        else:
            while cur < N:
                add_bridge(cur, bridge_degree, nodes=nodes, edges=edges, logits=logits)
                cur += 1
                labels += [0]
        edge_labels += [0]*(len(edges) - st_links)
    return nodes, edges, labels, edge_labels, cmap


def generate_fixed_clique_graph(N: int, 
                                clique_degree: int,
                                bridge_degree: int, 
                                cc: float,
                                new_bridge: bool=True,
                                fill: bool=True,
                                structure_ratio: float = 1.0):
    '''
    cc - proportion of clique nodes. Cb= 1 - cc. c belongs to (0; 1).
    K -  number of clique nodes K = N * cc
    M -  number of bridge nodes M = N *( 1 - cc)
    bridge nodes only connect to clique nodes. 
    Returns:
       - (nodes, edges, labels) tuple
    '''
    K = int(N * cc)
    number_of_cliques = int(K / clique_degree)
    
    nodes = list(range(N))
    edges = []
    edge_labels = []
    cur = 0
    cmap = { clique_degree: number_of_cliques}
    clique_sizes = [clique_degree]
    if fill:
        M = clique_degree**2
        leftOver = K - M*number_of_cliques
        for l in range(clique_degree-1,1,-1):
            m = l*l
            if leftOver > m:
                if l in cmap:
                    val = cmap[l] + 1
                else:
                    val = 1
                leftOver -= l*l
                cmap[l] = val
    for j in cmap.keys():
        for i in range(cmap[j]):
            n, e = add_clique(K=j, start_index=cur, 
                              add_nodes=False, nodes=nodes, 
                              edges=edges, ratio=structure_ratio)
            cur += j
    print("Number of cliques: ", cmap)
    st_links = len(edges)
    edge_labels += [1]*st_links
    logits = [1.0 for _ in range(cur)]
    ## Secondly, Add bridge nodes
    label = cur
    if new_bridge:
        edges = generate_bridges(nodes, edges, cur, N,  bridge_degree)
    else:
        while cur < N:
            add_bridge(cur, bridge_degree, nodes=nodes, edges=edges, logits=logits)
            cur += 1
    labels = [1]*label + [0]*(N - label)
    edge_labels += [0]*(len(edges) - st_links)
    return nodes, edges, labels, edge_labels, cmap

def getNumberOfLinksIn2dLattice(K: int, M: int):
    """
    Returns number of edges for the KxM lattice.
    """
    g = nx.grid_2d_graph(K, M)
    return len([1 for _ in g.edges])

def computeIdealAUCFull(N: int, 
                        ratio: float, 
                        k: int, 
                        m: int, 
                        bridge_degree: int,
                        is_clique: bool,
                        clique_prob: float,
                        diagonals: int = 0,
                        ) -> float:
    """
    Params:
    N - number of the verticles
    ratio - ratio between total and lattice nodes
    k - horizontal size of the lattice
    m - vertical size of the lattice
    bridge_degree - bridge degree
    diagonals - how many diagonals do we need ot add. 0.1.2.3.
    """
    # Real realization of number of Structural nodes:
    if is_clique and k != m:
        raise ValueError("K != M for cliques case.")
    if is_clique:
        nodes_vv = int(N*ratio * k / k) 
        V = int(int(N*ratio * k / k) * (k - 1)/2)
    else:
        nodes_vv = int(N*ratio*(k*m) / (k*m)) 
        gamma = 0
        if diagonals == 1 or diagonals == 2:
            gamma = 1
        elif diagonals == 3:
            gamma = 2
        V = int(N*ratio * (getNumberOfLinksIn2dLattice(k, m) + gamma * k*m) / (k*m) )
        # OLD: V = int(N*ratio * getNumberOfLinksIn2dLattice(k, m) / (k*m))  
    W = int((N - nodes_vv) * bridge_degree)
    Y = int(nodes_vv**2 / 2) - V
    X = int((N - nodes_vv)*nodes_vv) - W
    Z = int((N - nodes_vv)**2 / 2)
    T = int(N*N/2)
    idealAUC =  (1 - 0.5 * (1.0 - V / (V + W)) * X / (X + Y + Z) )
    print("Ideal N = %d Structural Nodes: %d" % (N, nodes_vv))
    return idealAUC, T, V, W, X, Y, Z

def computeIdealAUC(N: int, 
                    ratio: float, 
                    k: int, 
                    bridge_degree: int, 
                    is_clique: bool,
                    clique_prob: float = 1.0,
                    diagonals: int = 0) -> float:
    return computeIdealAUCFull(N=N, ratio=ratio, k=k, m=k, bridge_degree=bridge_degree, 
                               is_clique=is_clique, clique_prob=clique_prob, diagonals=diagonals)

def computeIdealiSBM(N: int, ratio: float, 
                     k: int, 
                     bridge_degree: int, 
                     is_clique: bool,
                     diagonals: int = 0,
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
        nodes_vv = int(N*ratio)
        gamma = 0
        if diagonals == 1 or diagonals == 2:
            gamma = 1
        elif diagonals == 3:
            gamma = 2
        V = int( N*ratio * (getNumberOfLinksIn2dLattice(k, k) + gamma * k*k) / (k*k))
        print("Gamma : ", gamma, " V = ", V)
        B = int(N*ratio*k*k*(k*k - 1)/(2*k*k)) # int(N*ratio*k*k*(k*k - 1)/(2*k*k))
        Q = int(nodes_vv * (nodes_vv - 1) / 2 - B)
    #assert abs(nodes_vv*nodes_vv/2 - B - Q) < 1000, "Q and B and N should follow rules: %f %f %f" % (abs(nodes_vv*nodes_vv/2 - B - Q), Q, B)
    if abs(nodes_vv*nodes_vv/2 - B - Q) > 200:
        print("DEVIATION : ", abs(nodes_vv*nodes_vv/2 - B - Q))
    print("Lattice links ",  getNumberOfLinksIn2dLattice(k, k) , nodes_vv*nodes_vv/2 - B, Q)
    print("Node_vv ", nodes_vv)
    # Z = int((N - nodes_vv)**2 / 2) - old
    Z = int((N - nodes_vv)*(N - nodes_vv - 1) / 2)
    W = (N - nodes_vv)*bridge_degree
    G = int(N*ratio*N*(1 - ratio) + 0.0000001)
    SS = B - V
    BS = G - W
    TS = Q + Z
    AS = BS + SS + TS
    Pp = bridge_degree / int(N * ratio)
    Pq = 2 * getNumberOfLinksIn2dLattice(k, k) / (k*k*(k*k - 1))
    print(' N = ', N)
    print('P = ', Pp, ' Q = ', Pq, ' Z = ', Z, ' B =', B, ' V = ', V , ' Q = ', Q, ' G = ', G, 'W = ', W)
    print('SS = ', SS, ' AS = ', AS, ' SS/AS = ', SS/AS, ' BS / AS', BS / AS)
    print('V = ', V, ' W = ', W, ' V/(V + W) = ', V/(V + W))
    if (N*(N - 1)) / 2 != B + Q + Z + G:
        print('FAILED !!! CHECK: ', (N*(N - 1)) / 2, B + Q + Z + G, ' must be equal')

    if N*ratio* (N*ratio - 1) / 2 != B+Q:
        print('FAILED !!! CHECK: ', N*ratio*(N*ratio - 1) / 2, B + Q, ' equal)must be')
    if Pq > Pp:
        print("Q > P")
        M = V / (V + W)
        isbm = 1 - (1 - M)*SS/AS - 0.5*(1 - M)*BS/AS - 0.5*M*SS/AS
        isbmt =  1 - 0.5 * (( 2*SS + BS )/(SS + BS + TS) - (V/(V+W))*((SS + BS) / (SS + BS +TS)))
        if abs(isbm - isbmt) > 1.0:
            print("DEVIATION in iSBM")
    else:
        print("P > Q")
        M = W / (V + W)
        isbm = 1 - (1 - M)*BS/AS - 0.5*(1 - M)*SS/AS - 0.5*M*BS/AS
    return isbm

    
def generate_graph(folder: str, 
                   *,
                   file_prefix: str,
                   N: int,
                   k: int,
                   ratioC: float,
                   bridge_degree: int,
                   diagonals: int = 1,
                   structure_ratio: float = 1.0,
                   graphs_to_generate: int = 10,
                   forcelink = False,
                   barabassi = False,
                   is_cliques: bool = False,
                   align_bridges = True,
                   new_bridge: bool = False,
                   graph_name_N: bool = False,
                  ):
    """Generates graph with given parameters."""
    print('Generating K=', k, ' Ratio: ', ratioC, ' B = ', bridge_degree, 'N = ', N)
    prefix = "%s%d" % (file_prefix, 0)
    if True:
        for c in range(graphs_to_generate):
            gp = GraphParams(N=N, 
                             cdegree=k,
                             bdegree=bridge_degree,
                             ratio=ratioC,
                             barabassi=barabassi,
                             clique=is_cliques,
                             prefix=prefix
                            )
            gp['structure_ratio'] = structure_ratio
            connected = False
            connected_tolerance = 10000 # Was only 5
            while not connected:
                st = time.monotonic()
                if is_cliques:
                    n, e, l, le, cmap = generate_fixed_clique_graph(N,
                                                              clique_degree=k,
                                                              bridge_degree=bridge_degree,
                                                              cc=ratioC, new_bridge=new_bridge,
                                                              structure_ratio=structure_ratio)
                else:
                    n, e, l, le, cmap = generate_fixed_lattice_graph(N, 
                                                               clique_degree=k, 
                                                               bridge_degree=bridge_degree, 
                                                               cc = ratioC,
                                                               bashrabassi=barabassi,
                                                               new_bridge=new_bridge,
                                                               forcelink=forcelink,
                                                               diags=diagonals,
                                                               align_bridges=align_bridges)
                G = nx.Graph()
                edge_labels = le.copy()
                G.add_nodes_from(n)
                G.add_edges_from(e)
                connected = nx.algorithms.components.is_connected(G)
                if not connected:
                    if connected_tolerance % 100 == 0:
                        print("\nConnected Graph: ", nx.algorithms.components.is_connected(G), " R=", ratioC, " Tol: ", connected_tolerance)
                    else:
                        print(".", end="")
                else:
                    print("Connected Graph: ", nx.algorithms.components.is_connected(G), " R=", ratioC, " Tol: ", connected_tolerance)
                if not connected:
                    connected_tolerance -= 1
                    if connected_tolerance < 0:
                        break
                    continue
            if not connected:
                print("Skipping the Graph parameters.")
                continue
            print("Structural links :%d Bridge Links: %d" %(sum(le), len(le) - sum(le)) )
            print("Structural nodes :%d Bridge nodes: %d" %(sum(l), len(l) - sum(l)) )
            pos = sample_random_positive_edges(n, e, edge_labels, 20)
            print("Positive ", len(pos), " from ", len(e))
            pos = set(pos)
            edg2 = list(set(e) - set(pos)) 
            not_edg = sample_negative_edges(n, e, len(e) + 2*len(pos))   
            not_edg = set(not_edg) - set(pos)
            not_edg = list(set(not_edg) - set(edg2))
            assert len(set(not_edg).intersection(pos)) == 0
            assert len(set(edg2).intersection(pos)) == 0
            assert len(set(edg2).intersection(not_edg)) == 0
            print("Found positives:", len(pos), " and negatives: ", len(not_edg))
            ps, ng = list(pos), list(not_edg[len(edg2):len(edg2) + len(pos)])
            test_labels = np.hstack([np.zeros(len(ng)), np.ones(len(ps))])
            test_edg = ng + ps
            eLen = len(e)
            gp['cmap'] = cmap
            print("Structure: ", cmap)
            gp['diagonals'] = diagonals
            auc,t,v,w,x,y,z  = computeIdealAUC(N, ratio=ratioC, 
                                               k=k, 
                                               bridge_degree=bridge_degree,
                                               is_clique=is_cliques)
            da = {"nodes": n, 
                    "edges": edg2, 
                    "labels": l, 
                    "edge_labels": edge_labels,
                    "positives": pos,
                    "negatives": not_edg,
                    "params": gp}
            print("Ideal: ", auc, t, v, w, x, y, z)
            emperical_auc, emp_d = calculate_perfect_auc(test_edg, test_labels, gp)
            gp.ideal_auc = auc
            gp.emperical_auc = emperical_auc
            gp['ideal_auc'] = auc
            gp['emperical_auc'] = emperical_auc
            gp['isbm_auc'] = computeIdealiSBM(N, ratio=ratioC, 
                                               k=k, 
                                               bridge_degree=bridge_degree,
                                               is_clique=is_cliques)
            if graph_name_N:
                save_graph(n, edg2, l, edge_labels, pos, not_edg, gp, "%s_%d_%d" % (prefix, N, c), folder)
            else:
                save_graph(n, edg2, l, edge_labels, pos, not_edg, gp, "%s_%d" % (prefix, c), folder)
            print("Idea AUC %.4f" % auc, " Emperical AUC: %.4f" % emperical_auc, " Ideal SBM: %.4f " % gp['isbm_auc'])
            
