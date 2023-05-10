from gensim.models import word2vec
import networkx as nx
import numpy as np
from tqdm import tqdm


def walkOneTime(g, start_node, walk_length):
    walk = [str(start_node)]
    for _ in tqdm(range(walk_length)):
        current_node = int(walk[-1])
        successors = list(g.successors(current_node))
        if len(successors) > 0:
            next_node = np.random.choice(successors, 1)
            walk.extend([str(n) for n in next_node])
        else:
            break
    return walk


def getDeepwalkSeqs(g, walk_length, num_walks):
    seqs = []
    for _ in range(num_walks):
        start_node = np.random.choice(g.nodes, 1)
        w = walkOneTime(g, start_node, walk_length)
        seqs.append(w)
    return seqs

    def deepwalk(g, dimensions=10, walk_length=80, num_walks=10, min_count=3):
        seqs = getDeepwalkSeqs(g, walk_length, num_walks)
        model = word2vec.Word2Vec(seqs, vector_size=dimensions, min_count=3)

if __name__ == '__main__':
    g = nx.fast_gnp_random_graph(n=100, p=0.5, directed=True)
    seqs = []
    for _ in tqdm(range(100)):
        start_node = np.random.choice(g.nodes)
        w = walkOneTime(g, start_node, 100)
        seqs.append(w)

    model = deepwalk(g, dimensions=10, walk_length=20, num_walks=100, min_count=3)
