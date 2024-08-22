import time
import tensorflow as tf
import pandas as pd
import pickle 
import stellargraph as sg

from stellargraph import StellarGraph, datasets
from stellargraph.data import EdgeSplitter, BiasedRandomWalk, UnsupervisedSampler
from stellargraph.mapper import GraphSAGELinkGenerator, GraphSAGENodeGenerator
from stellargraph.layer import GraphSAGE, Node2Vec, link_classification
from sklearn.metrics import roc_auc_score
import networkx as nx
import random
import numpy as np
from matplotlib import pyplot as plt

NODE_DIM = 16

def get_graph_data(fname):
    data = None
    with open(fname, "rb") as fr:
        data = pickle.load(fr)
        fr.close()
    return data

def get_name(prefix, N, i, j):
  if N:
    fname = "./data/graph_%s%d_%d_%d.pkl" % (prefix, i, N, j)
  else:
    fname = "./data/graph_%s%d_%d.pkl" % (prefix, i,  j) # - new, No N
  return fname

def create_links(nodes, edges):
    square_edges = pd.DataFrame(
        {"source": [x[0] for x in edges], "target": [x[1] for x in edges]}
    )
    return square_edges

def graph_from_data(data, embeddings, node_dim, full:bool = True):
    nodes = data['nodes']
    fmap = {'w%d'% i:list(embeddings[:, i]) for i in range(node_dim)}  
    visible_edges = data['edges']
    if full:
        visible_edges += list(data['positives'])
    links = create_links(nodes, visible_edges)
    snodes = pd.DataFrame(fmap, index=nodes)
    graph = sg.StellarGraph(snodes, links)
    return graph

def order_graph(nod: list, 
                edg: list, 
                cmap: dict,
                labels: list, 
                max_node: int,
                row: bool = True):
    """Assume for simplicity cmap has only one block type."""
    vs = set() # node 
    es = [] # edges
    pos = []
    x = 0
    y = 0
    for i in cmap.items():
        break
    K = i[0] 
    cl_node = K*K # Lattice 3x3
    block = []
    bp = 0
    for node in list(nod[:max_node]):
        if labels[node]:
            vs.add(node)
            pos.append([bp*K + x, y])
            block.append(bp)
            x += 1
            if x >= K:
                y += 1
                x = 0
            if node and (node+1) % cl_node == 0:
                bp += 1
                if row:
                    y = 0 # ROW
                else:
                    y += 1
        else:
            y = K+1
            maxx = max([x[0] for x in pos])
            x = random.randint(0, maxx)
            vs.add(node)
            pos.append([x, y])
            block.append(bp)
    for e in edg:
        if e[0] in vs and e[1] in vs:
            es.append(e)
    maxx = max([x[0] for x in pos])
    maxy = max([x[1] for x in pos])
    if row:
        pos = [(x[0]/maxx, x[1]/maxx) for x in pos]
    else:
        pos = [(x[0]/maxx, x[1]/maxy) for x in pos]
    # TO draw it well 
    #vs.add(node+1)
    #pos += [(1,1)]
    #block.append(bp+1)
    return list(vs), es, block, pos

def add_sym(edges, sim=False):
    nedge = []
    for e in edges:
        nedge.append(e)
        if sim:
            nedge.append((e[1], e[0]))
    return nedge

def draw_first_nodes(nod, edg, pos, colors, max_node:int):
    G = nx.Graph(directed=False)
    nodesnx = [x for x in nod[:max_node]]
    alledges = add_sym(edg, False)
    alle = []
    for e in alledges:
        if e[0] < max_node and e[1] < max_node:
            alle.append(e)

    G.add_nodes_from(nodesnx)
    G.add_edges_from(alle)
    print(nodesnx)
    nx.draw(G, pos=pos, node_color=colors, node_size=100, style='--', edge_color='silver')
    plt.show()

def print_graph(nod, 
                edg, 
                *, 
                cmap, 
                all_colours, 
                labels, 
                node_labels,
                max_node:int=192):
    ns, es, blocks, poss = order_graph(nod, edg, cmap, labels, max_node)
    ns += [ns[-1]+1]
    es += [(0, ns[-1]+1)]
    if isinstance(node_labels, np.ndarray):
        blocks = list(node_labels) + [max(blocks)+1]
    else:
        blocks += [max(blocks)+1]
    poss += [(0.0, 1.0)]
    print(len(poss), len(nod))
    colours = [all_colours[blocks[n]] for n in ns]
    fig = plt.figure(1, figsize=(15, 15))
    draw_first_nodes(ns, es, poss, colours, max_node=max_node+1)


def get_node_inits(node_len, node_dim: int = NODE_DIM):
    initializer = tf.keras.initializers.GlorotNormal(seed=42)
    values = initializer(shape=(node_len, node_dim)).numpy()
    return values


def graph_from_data(data, embeddings, node_dim, full:bool = True):
    nodes = data['nodes']
    fmap = {'w%d'% i:list(embeddings[:, i]) for i in range(node_dim)}  
    visible_edges = data['edges']
    if full:
        visible_edges += list(data['positives'])
    links = create_links(nodes, visible_edges)
    snodes = pd.DataFrame(fmap, index=nodes)
    graph = sg.StellarGraph(snodes, links)
    return graph

def create_biased_random_walker(graph, walk_num, walk_length):
    # parameter settings for "p" and "q":
    p = 1.0
    q = 1.0
    return BiasedRandomWalk(graph, n=walk_num, length=walk_length, p=p, q=q)

def graphsage_embedding(graph, 
                        name: str, 
                        walk_number=20, 
                        walk_length=10,
                        batch_size=4,
                        dimensions=[16, 16],
                        num_samples = [10, 5]):

    print(f"Training GraphSAGE for '{name}':")

    graph_node_list = list(graph.nodes())

    # Create the biased random walker to generate random walks
    walker = create_biased_random_walker(graph, walk_number, walk_length)

    # Create the unsupervised sampler to sample (target, context) pairs from random walks
    unsupervised_samples = UnsupervisedSampler(graph, nodes=graph_node_list, walker=walker)

    # Define a GraphSAGE training generator, which generates batches of training pairs
    generator = GraphSAGELinkGenerator(graph, batch_size, num_samples)

    # Create the GraphSAGE model
    graphsage = GraphSAGE(
        layer_sizes=dimensions,
        generator=generator,
        bias=True,
        dropout=0.1,
        normalize="l2",
    )

    # Build the model and expose input and output sockets of GraphSAGE, for node pair inputs
    x_inp, x_out = graphsage.in_out_tensors()

    # Use the link_classification function to generate the output of the GraphSAGE model
    prediction = link_classification(
        output_dim=1, output_act="sigmoid", edge_embedding_method="ip"
    )(x_out)

    # Stack the GraphSAGE encoder and prediction layer into a Keras model, and specify the loss
    model = tf.keras.Model(inputs=x_inp, outputs=prediction)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=tf.keras.losses.binary_crossentropy,
        metrics=[tf.keras.metrics.binary_accuracy],
    )
    return model, generator, unsupervised_samples

def compute_auc(predicted, test_labels):
    res = roc_auc_score(y_true=test_labels, y_score=predicted)
    return res

def join_cap_data(X, y, cap: int):
    if len(X) != len(y):
        raise ValueError("Misalligned data")
    joint = list(zip(X, y))
    random.shuffle(joint)
    data = joint[:cap]
    newX = [x[0] for x in data]
    newY = [x[1] for x in data]
    return newX, newY

def train_graphsage_with_data(data,
    epochs:int = 12,
    batch:int = 128,
    num_samples:list = [10, 5],
    walks: int  = 20,
    workers:int = 8
):
    st = time.monotonic()
    n = data['params']['N']
    n = len(data['nodes'])
    emb = get_node_inits(n)
    visible_edges = list(data['edges'])
    pos = list(data['positives'])
    neg = list(data['negatives'])
    test_labels = [1]*len(pos) + [0]*len(pos)
    test = pos + neg[:len(pos)]
    train_labels = [1]*len(visible_edges) + [0]*len(visible_edges)
    train = visible_edges + neg[len(pos):len(pos) + len(visible_edges)]
        
    fullgraph = graph_from_data(data, emb, node_dim=NODE_DIM, full=True)
    traingraph = graph_from_data(data, emb, node_dim=NODE_DIM, full=False)
        
    full_gen = GraphSAGELinkGenerator(fullgraph, batch, num_samples)
    train_gen = GraphSAGELinkGenerator(traingraph, batch, num_samples)
    print("Train size: %d" % len(train))
    train, train_labels = join_cap_data(train, train_labels, 1000000)
        #if len(train) > 25000:
        #    train, train_labels = join_cap_data(train, train_labels, 25000)
        #if len(test) > 15000:
        #    test, test_labels = join_cap_data(test, test_labels, 15000)
    test_flow = full_gen.flow(test, test_labels)
    train_flow = train_gen.flow(train, train_labels)
    model, generator, unsupervised_samples = graphsage_embedding(
            traingraph, 
            name='graphemb',
            batch_size=batch,
            walk_number=walks,
            num_samples=num_samples)
    print("Starting training!")

    model.fit(
                generator.flow(unsupervised_samples),
                epochs=epochs,
                use_multiprocessing=True,
                workers=workers,
                shuffle=True,
                verbose=2
            )
    print("One training took: %d", time.monotonic() - st)
    probs = model.predict(test_flow, verbose=1)
    ypred = list(np.squeeze(probs, axis=1))
    auc = compute_auc(predicted=ypred, test_labels=test_labels)
    del model
    tf.keras.backend.clear_session()
    return auc, time.monotonic() - st

def train_graphsage(epochs: int = 12,
                    batch: int = 128,
                    num_samples: list = [10, 5],
                    prefix: str = 'faab',
                    walks: int = 20,
                    workers: int = 8,
                    N:int = 3200,
                    i:int = 0,
                    j:int = 0):
    fname = get_name(prefix, N, i, j)
    data = get_graph_data(fname)
    return train_graphsage_with_data(data,
                                     epochs=epochs,
                                     batch=batch,
                                     num_samples=num_samples,
                                     workers=workers,
                                     walks=walks)
