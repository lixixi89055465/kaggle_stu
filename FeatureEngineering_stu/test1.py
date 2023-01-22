import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import neighbors, datasets
# from sklearn.neighbors.nearest_centroid import NearestCentroid
from sklearn.neighbors._nearest_centroid import NearestCentroid

n_neighbors = 15

# 加载数据
iris = datasets.load_iris()
# print(iris)
# 二维可视化
X = iris.data[:, :2]
y = iris.target

cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

plt.figure()

plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold)
plt.xlim(X.min(), 8)
plt.ylim(y.min(), 6)
plt.title("3-Class classification")
plt.show()

# 可视化分类器及数据
h = .02

cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])

clf = NearestCentroid()
clf.fit(X, y)

# 计算每个特征向量的最大值和最小值
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

# 可视化分类器
Z = Z.reshape(xx.shape)
plt.figure()
plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

# 可视化数据
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold)
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.title("3-Class classification")

plt.show()

