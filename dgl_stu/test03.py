from collections import Counter

import dgl

g = dgl.heterograph({
    ('user', 'follows', 'user'): [(0, 1), (1, 2)],
    ('user', 'plays', 'game'): [(0, 0), (1, 0), (1, 1), (2, 1)],
    ('developer', 'develops', 'game'): [(0, 0), (1, 1)],
})

print(g.number_of_nodes('user'))
print(g.number_of_edges('plays'))
print(g.out_degrees(etype='develops'))
print(g.in_degrees(0, etype='develops'))
print(g.in_degrees(0, etype='plays'))

print("0" * 100)
print(g['develops'].in_edges(0))
print(g['plays'].in_edges(1))
print("1" * 100)
print(g['follows'].number_of_nodes())
print(g['plays'].number_of_nodes())
print(g['plays'].number_of_edges())
