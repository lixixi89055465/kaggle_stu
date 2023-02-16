import numpy as np

print(np.c_[np.array([1, 2, 3]), np.array([4, 5, 6])])
print(np.c_[np.array([[1, 2, 3]]), 0, 0, np.array([[4, 5, 6]])])
