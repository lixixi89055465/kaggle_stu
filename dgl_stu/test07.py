import dhg

hg = dhg.Hypergraph(5, [(0, 1, 2), (2, 3), (0, 4)])
print(hg)
print("0" * 100)
print(hg.e)
print("1" * 100)
print(hg.H.to_dense())

print("2" * 100)
hg = dhg.Hypergraph(5, [(0, 2, 1), (2, 3), (0, 4)])
print(hg.e)
print("3" * 100)
print(hg.H.to_dense())

hg = dhg.Hypergraph(5, [(1, 0, 2), (2, 3), (0, 4)])
print(hg.e)
print(hg.H.to_dense())

print("4" * 100)
hg = dhg.Hypergraph(5, [(0, 1, 2), (2, 3), (2, 3), (0, 4)], merge_op='mean')
print(hg.e)
hg.add_hyperedges([(0, 2, 1), (0, 4)], merge_op='sum')
print(hg.e)
print("5" * 100)

hg = dhg.Hypergraph(5, [(0, 1, 2), (2, 3), (2, 3), (0, 4)], merge_op="sum")
print(hg.e)
print("6" * 100)
hg.add_hyperedges([(0, 2, 1), (0, 4)], merge_op="mean")
print(hg.e)
print("7" * 100)
import torch

X = torch.tensor([[0.0658, 0.3191, 0.0204, 0.6955],
                  [0.1144, 0.7131, 0.3643, 0.4707],
                  [0.2250, 0.0620, 0.0379, 0.2848],
                  [0.0619, 0.4898, 0.9368, 0.7433],
                  [0.5380, 0.3119, 0.6462, 0.4311],
                  [0.2514, 0.9237, 0.8502, 0.7592],
                  [0.9482, 0.6812, 0.0503, 0.4596],
                  [0.2652, 0.3859, 0.8645, 0.7619],
                  [0.4683, 0.8260, 0.9798, 0.2933],
                  [0.6308, 0.1469, 0.0304, 0.2073]])
hg = dhg.Hypergraph.from_feature_kNN(X, k=3)
print(hg)
print("8" * 100)
print(hg.e)
print("9" * 100)
print(hg.H.to_dense())
