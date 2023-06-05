import dhg

g = dhg.DiGraph(5, [(0, 3), (2, 4), (4, 2), (3, 1)])
print(g)
print("0" * 100)
print(g.e)
print("1" * 100)
print(g.A.to_dense())

print("2" * 100)
g = dhg.DiGraph.from_adj_list(5, [(0, 3, 4), (2, 1, 3), (3, 0)])
print(g)
print(g.e)
print("3" * 100)
print(g.A.to_dense())

import torch

X = torch.tensor(([[0.6460, 0.0247],
                   [0.9853, 0.2172],
                   [0.7791, 0.4780],
                   [0.0092, 0.4685],
                   [0.9049, 0.6371]]))
g = dhg.DiGraph.from_feature_kNN(X, k=2)
print(g)
print("4" * 100)
print(g.e)
print("5" * 100)
print(g.A.to_dense())
