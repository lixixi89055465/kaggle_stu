import numpy as np
from scipy.sparse import csr_matrix

arr = np.array([
    0, 0, 0, 0, 0, 1, 1, 0, 2
])
print(csr_matrix(arr))
print('1' * 100)
arr = np.array([
    [0, 0, 0],
    [0, 0, 1],
    [1, 0, 2],
])
print(csr_matrix(arr))
print(csr_matrix(arr).data)
print('2' * 100)
print(csr_matrix(arr).count_nonzero())
print('3' * 100)
mat = csr_matrix(arr)
mat.eliminate_zeros()
print(mat)
print('4' * 100)
mat=csr_matrix(arr)
mat.sum_duplicates()
print(mat)
print('5' * 100)
newarr=csr_matrix(arr).tocsc()
print(newarr)
