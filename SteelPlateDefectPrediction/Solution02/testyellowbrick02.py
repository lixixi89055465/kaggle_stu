# -*- coding: utf-8 -*-
# @Time : 2024/4/13 17:53
# @Author : nanji
# @Site : https://zhuanlan.zhihu.com/p/396665902
# @File : testyellowbrick02.py
# @Software: PyCharm 
# @Comment : 
import pandas as pd
from sklearn.cluster import KMeans
from yellowbrick.cluster.elbow import kelbow_visualizer
from sklearn.datasets import load_iris

data_df = load_iris()['data']
print(data_df)
print('0' * 100)
# print(data_df.head())
# X = data_df[["sepal_length","sepal_width","petal_length","petal_width"]]
X = data_df
oz = kelbow_visualizer(KMeans(random_state=1), X, k=(2, 10))
k = oz.elbow_value_
print(f"最佳的K值是{k}")
print(f"elbow_score_值是{oz.elbow_score_}")
kelbow_visualizer(KMeans(random_state=1), X, k=(2, 10), metric='distortion')
print('1' * 100)
kelbow_visualizer(KMeans(random_state=1), X, k=(2, 10), metric='silhouette')
print('2' * 100)
kelbow_visualizer(KMeans(random_state=1), X, k=(2, 10), metric='calinski_harabasz')
