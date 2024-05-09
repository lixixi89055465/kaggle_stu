# -*- coding: utf-8 -*-
# @Time : 2024/5/8 23:51
# @Author : nanji
# @Site : https://blog.csdn.net/qq_14997473/article/details/96840513
# @File : testsilhouette_score.py
# @Software: PyCharm 
# @Comment : 
import warnings

warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn import metrics

estimator = KMeans(n_clusters=10, random_state=777)
kmean_data_tf = ""
estimator.fit(kmean_data_tf)
r1 = pd.Series(estimator.labels_).value_counts()
r2 = pd.DataFrame(estimator.cluster_centers_)
r = pd.concat([r2, r1], axis=1)
r.columns = list(kmean_data_tf.columns) + [u'类别数据']
print(r)

print('轮廓系数:', metrics.silhouette_score(kmean_data_tf, estimator.labels_, metrics='euclidean'))

