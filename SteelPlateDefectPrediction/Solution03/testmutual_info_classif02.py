# -*- coding: utf-8 -*-
# @Time : 2024/5/21 22:09
# @Author : nanji
# @Site :https://blog.csdn.net/iizhuzhu/article/details/105031532
# @File : testmutual_info_classif02.py
# @Software: PyCharm 
# @Comment :

# from sklearn.datasets import load_iris
# from sklearn.feature_selection import SelectKBest
# from sklearn.feature_selection import chi2
#
# X, y = load_iris(return_X_y=True)
# X_new = SelectKBest(chi2, k=2).fit_transform(X, y)
# print(X_new.shape)


import numpy as np
from sklearn.metrics import mutual_info_score, \
	normalized_mutual_info_score, \
	adjusted_mutual_info_score

rng = np.random.RandomState(0)
y1 = rng.randint(0, 10, size=100)
y2 = rng.randint(0, 10, size=100)
print(y1)
y2[:20] = y1[:20]
print('MI')
print(mutual_info_score(y1, y2), mutual_info_score(y1 % 3, y2 % 3))
print('1'*100)

print('NMI')
print(normalized_mutual_info_score(y1,y2),normalized_mutual_info_score(y1%3,y2%3))

print('AMI')
print(adjusted_mutual_info_score(y1,y2),adjusted_mutual_info_score(y1%3,y2%3))

