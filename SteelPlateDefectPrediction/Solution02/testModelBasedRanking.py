# -*- coding: utf-8 -*-
# @Time : 2024/5/1 20:45
# @Author : nanji
# @Site :https://zhuanlan.zhihu.com/p/141506312 
# @File : testModelBasedRanking.py
# @Software: PyCharm 
# @Comment : 
from sklearn.model_selection import cross_val_score, ShuffleSplit
from sklearn.datasets import load_boston
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import warnings

warnings.filterwarnings('ignore')
boston = load_boston()
X = boston['data']
Y = boston['target']
names = boston['feature_names']
print('0' * 100)
print(names)
rf = RandomForestRegressor(n_estimators=20, max_depth=4)
scores = []
for i in range(X.shape[1]):
	score = cross_val_score(
		rf,\
		X[:, i:i + 1], \
		Y, \
		scoring='r2', \
		cv=ShuffleSplit(len(X), 3, .3))
	scores.append((format(np.mean(score), '.3f'), names[i]))
print('0' * 100)
print(sorted(scores, reverse=True))
