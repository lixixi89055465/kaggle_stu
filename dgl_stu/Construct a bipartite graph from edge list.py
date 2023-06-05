import dhg

g = dhg.BiGraph(5, 4, [(0, 3), (4, 2), (1, 1), (2, 0)])
print(g)
print(g.e)
print("0"*100)
print(g.B.to_dense())
print("1"*100)
print(g.A.to_dense())
