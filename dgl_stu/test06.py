import dhg.random as dr

g = dr.graph_Gnm(10, 20)
print(g)

print("0"*100)
g=dr.graph_Gnp(10,0.5)
print(g)
print("1"*100)
g=dr.graph_Gnp(10,0.5)
print(g)
print("2"*100)
g=dr.graph_Gnp_fast(10,0.5)
print(g)