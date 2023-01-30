import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import neighbors, datasets
# from sklearn.neighbors.nearest_centroid import NearestCentroid
from sklearn.neighbors._nearest_centroid import NearestCentroid
import pandas as pd

df = pd.DataFrame({
    'a': [1, 2] * 3,
    'b': [True, False] * 3,
    'c': [1.0, 2.0] * 3,
    'd': ['1', '2'] * 3
})
print(df.info())
# print(df.select_dtypes(include=['int64']))
# print(df.select_dtypes(include='object'))
# print(df.select_dtypes(exclude="bool"))
print('1' * 100)
print(list(df.select_dtypes(include='int64').columns))
print('2' * 100)
numerical_fea = list(df.select_dtypes(include=['int64', 'float64']).columns)
print(numerical_fea)
