import dhg

g = dhg.Graph(5, [(0, 1), (0, 2), (1, 2), (3, 4)])
print(g)
print("0" * 100)
print(g.v)
print("1" * 100)
print(g.e)
print("2" * 100)
print(g.e_both_side)

print("3" * 100)
print(g.A.to_dense())

print("4" * 100)
g = dhg.Graph(5, [(0, 1), (0, 2), (2, 0), (3, 4)])
print(g.e)
g.add_edges([(0, 1), (2, 3)])
print("5"*100)
print(g.e)
