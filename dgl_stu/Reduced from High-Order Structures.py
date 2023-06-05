import dhg

hg = dhg.Hypergraph(5, [(0, 1, 2), (1, 3, 2), (1, 2), (0, 3, 4)])
print(hg.e)

print("0" * 100)
print(hg.H.to_dense())

print("1" * 100)
g, v_mask = dhg.Graph.from_hypergraph_star(hg)
print("2" * 100)
print(g)
print(g.e[0])
print("3" * 100)
print(v_mask)

g = dhg.Graph.from_hypergraph_clique(hg)
print(g)
print("4" * 100)
print(g.e)
print("5" * 100)
print(g.A.to_dense())
print("6" * 100)
import torch

X = torch.tensor(([[0.6460, 0.0247],
                   [0.9853, 0.2172],
                   [0.7791, 0.4780],
                   [0.0092, 0.4685],
                   [0.9049, 0.6371]]))
g = dhg.Graph.from_hypergraph_hypergcn(hg, X)
print("7" * 100)
print(g)
print("8" * 100)
print(g.e)
print("9" * 100)
print(g.A.to_dense())

