from collections import Counter

import dgl

GG = dgl.heterograph({
    ('user', 'watches', 'movie'): [(0, 1), (1, 0), (1, 1)],
    ('user', 'watches', 'TV'): [(0, 0), (1, 1)]
})
print(GG)

print("3" * 100)
print(GG.number_of_edges(('user', 'watches', 'movie')))
print(GG.number_of_edges(('user', 'watches', 'TV')))
