# -*- coding: utf-8 -*-
# @Time : 2024/4/15 22:02
# @Author : nanji
# @Site : https://blog.csdn.net/tMb8Z9Vdm66wH68VX1/article/details/130177587
# @File : testnan_euclidean_distances.py
# @Software: PyCharm 
# @Comment :
import numpy as np
from sklearn.metrics.pairwise import nan_euclidean_distances

# X=[[3,np.nan,5]]
# Y=[[1,0,0]]
# r1=nan_euclidean_distances(X,Y)
# print(r1)

# X = [[3, np.nan, 5], [1, 0, 0]]
# r2 = nan_euclidean_distances(X, X)
# print(r2)

X = [[3, np.nan, 5], [1, 0, 0], [3, 3, 3]]
r3 = nan_euclidean_distances(X, X)
print(r3)
