import numpy as np

a1 = np.array([1, 2, 3, 34, 4])
print(id(a1))
a2=a1.ravel()
print(id(a2))
a3=a1.flatten()
print(id(a3))
