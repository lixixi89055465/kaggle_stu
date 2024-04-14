# -*- coding: utf-8 -*-
# @Time : 2024/4/13 21:59
# @Author : nanji
# @Site : https://blog.csdn.net/LuohenYJ/article/details/107575950
# @File : testyellowbrick04.py
# @Software: PyCharm 
# @Comment : 
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

from yellowbrick.cluster import KElbowVisualizer

# Generate synthetic dataset with 8 random clusters
# 建立具有8个随机点簇中心的数据
X, y = make_blobs(n_samples=1000, n_features=12, centers=8, random_state=42)

# Instantiate the clustering model and visualizer
model = KMeans()
# 可视化
visualizer = KElbowVisualizer(model, k=(4,12))

visualizer.fit(X)        # Fit the data to the visualizer
visualizer.show();        # Finalize and render the figure
