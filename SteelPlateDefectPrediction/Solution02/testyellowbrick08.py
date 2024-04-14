# -*- coding: utf-8 -*-
# @Time : 2024/4/14 11:24
# @Author : nanji
# @Site : https://blog.csdn.net/wiborgite/article/details/96455545
# @File : testyellowbrick08.py
# @Software: PyCharm 
# @Comment : 

import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

from yellowbrick.cluster import KElbowVisualizer, SilhouetteVisualizer

mpl.rcParams["figure.figsize"] = (9, 6)

# Generate synthetic dataset with 8 blobs
X, y = make_blobs(n_samples=1000, n_features=12, centers=8, shuffle=True, random_state=42)

# 数据样本1000
# 样本的特征数量12
# 聚类数量8

# Instantiate the clustering model and visualizer
# model = KMeans()
# visualizer = KElbowVisualizer(model, k=(4, 12))
#
# visualizer.fit(X)  # Fit the data to the visualizer
# visualizer.poof()  # Draw/show/poof the data

# 横轴：K    左侧纵轴：失真分数（越小越好）  右侧纵轴：拟合时间（约小越好）
# 蓝色线为不同K时的失真分数，淡绿色为拟合时间
# Instantiate the clustering model and visualizer
model = KMeans(8)
visualizer = SilhouetteVisualizer(model)

visualizer.fit(X)  # Fit the data to the visualizer
visualizer.poof()  # Draw/show/poof the data

print('1'*100)
# Instantiate the clustering model and visualizer
model = KMeans(6)
visualizer = SilhouetteVisualizer(model)

visualizer.fit(X)  # Fit the data to the visualizer
visualizer.poof()  # Draw/show/poof the data