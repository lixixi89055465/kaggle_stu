from time import time  # 用于计算运行时间
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import offsetbox  # 定义图形box的格式
from sklearn import (manifold, datasets, decomposition, ensemble,
                     discriminant_analysis, random_projection)

digits = datasets.load_digits(n_class=6)
print(digits)
# 获取bunch中的data,target
print(digits.data)
print(digits.target)
digits = datasets.load_digits(n_class=6)
X = digits.data
y = digits.target
n_samples, n_features = X.shape
n_neighbors = 5
clf = manifold.LocallyLinearEmbedding(n_neighbors=n_neighbors, n_components=16, method='standard')
t0 = time()
X_lle = clf.fit_transform(X)
print(X_lle.shape)
