import pandas as pd
import numpy as np

arr = pd.arrays.SparseArray([0, 0, 1, 2])
print(arr)
print("1" * 100)
print(arr.astype(np.dtype('str')))
