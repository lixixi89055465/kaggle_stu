import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
# from sklearn.datasets.samples_generator import make_blobs
from sklearn.datasets._samples_generator import make_blobs
import pandas as pd

X, y = make_blobs(n_samples=10000, n_features=3,
                  centers=[[3, 3, 3], [0, 0, 0], [1, 1, 1], [2, 2, 2]],
                  cluster_std=[0.2, 0.1, 0.2, 0.2],

                  random_state=9)

fig = plt.figure()

ax = Axes3D(fig, rect=[0, 0, 1, 1], elev=30, azim=20)

plt.scatter(X[:, 0], X[:, 1], X[:, 2], marker='o')
# plt.show()
from sklearn.decomposition import PCA

pca = PCA(n_components=3)
pca.fit(X)
print('1' * 100)
print(pca.explained_variance_ratio_)
print('2' * 100)
print(pca.explained_variance_)

print('2' * 100)
pca = PCA(n_components=2)
pca.fit(X)
print(pca.explained_variance_ratio_)
print(pca.explained_variance_)
print('3' * 100)
X_New = pca.transform(X)
pca = PCA(n_components=0.95)
pca.fit(X)
print(pca.explained_variance_)
print(pca.explained_variance_ratio_)
pca = PCA(n_components=0.99)
pca.fit(X)
print(pca.explained_variance_)
print(pca.explained_variance_ratio_)
print("4" * 100)
pca = PCA(n_components='mle')
newX = pca.fit_transform(X)
print(pca.explained_variance_ratio_)
print(pca.explained_variance_)
print("5" * 100)
print(pca.components_)
print(pca.mean_)
print(pca.noise_variance_)
newXX = pca.inverse_transform(newX)
print(newXX.shape)
#
Xa=pca.transform(X)
# 将数据X转换成降维后的数据。当模型训练好后，对于新输入的数据，都可以用transform方法来降维。
print("6" * 100)
print(Xa.shape)
