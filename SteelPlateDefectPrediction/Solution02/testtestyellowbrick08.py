# -*- coding: utf-8 -*-
# @Time : 2024/4/14 9:57
# @Author : nanji
# @Site : https://blog.csdn.net/LuohenYJ/article/details/107575950
# @File : testtestyellowbrick08.py
# @Software: PyCharm 
# @Comment : 
# from sklearn.cluster import KMeans
#
# from yellowbrick.cluster import silhouette_visualizer
# from yellowbrick.datasets import load_credit
# # Load a clustering dataset
# X, y = load_credit()
# print(X.columns)
# X=X[(X['age']<=40)&(X['edu'].isin([1,2]))]
# print(X.shape)
# silhouette_visualizer(KMeans(5,random_state=42),X,colors='yellowbrick')
# from sklearn.cluster import KMeans
# from sklearn.datasets import make_blobs
#
# from yellowbrick.cluster import InterclusterDistance
# # Generate synthetic dataset with 12 random clusters
# X, y = make_blobs(n_samples=1000, n_features=12, centers=12, random_state=42)
#
# # Instantiate the clustering model and visualizer
# # 六个簇类
# model = KMeans(6)
# visualizer = InterclusterDistance(model)
#
# visualizer.fit(X)  # Fit the data to the visualizer|
# visualizer.show()  # Finalize and render the figure

from yellowbrick.datasets import load_nfl
from sklearn.cluster import MiniBatchKMeans
from yellowbrick.cluster import intercluster_distance


X, _ = load_nfl()
intercluster_distance(MiniBatchKMeans(5, random_state=777), X);


