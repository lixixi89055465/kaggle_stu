# -*- coding: utf-8 -*
import numpy as np

x_1 = np.array([1, 2, 3, 4, 5, 6]).reshape(2, 3)
x_2 = np.array([3, 2, 1, 8, 9, 6]).reshape(2, 3)
x_new = np.c_[x_1, x_2]
print("x_1 = \n", x_1)
print("x_2 = \n", x_2)
print("x_new = \n", x_new)

print("1" * 100)
x1 = np.array([[[1, 2, 3, 8], [1, 2, 3, 8], [1, 2, 3, 8]],
               [[1, 2, 3, 8], [1, 2, 3, 8], [1, 2, 3, 8]]])
x2 = np.array([[[0, 2, 3, 8], [1, 3, 5, 8], [1, 2, 3, 8]],
               [[1, 2, 9, 8], [9, 2, 3, 8], [1, 2, 3, 8]]])
print(x1.shape)
a = np.c_[x1, x2]
print(a, a.shape)
