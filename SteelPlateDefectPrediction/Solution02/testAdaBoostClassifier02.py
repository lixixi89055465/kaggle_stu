# -*- coding: utf-8 -*-
# @Time    : 2024/5/12 下午5:06
# @Author  : nanji
# @Site    :https://blog.csdn.net/gracejpw/article/details/102628310
# @File    : testAdaBoostClassifier02.py
# @Software: PyCharm 
# @Comment :
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_gaussian_quantiles

X1, y1 = make_gaussian_quantiles(cov=2.0, \
								 n_samples=500, \
								 n_features=2, \
								 n_classes=2, \
								 random_state=1)
X2, y2 = make_gaussian_quantiles(
	mean=(3, 3), \
	cov=1.5, \
	n_samples=400, \
	n_features=2, \
	n_classes=2, \
	random_state=1
)
X = np.concatenate((X1, X2))
y = np.concatenate((y1, -y2 + 1))
# plt.scatter(X[:, 0], X[:, 1], marker='.', c=y)
# plt.show()
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(
	np.arange(x_min, x_max, 0.02),
	np.arange(y_min, y_max, 0.02),
)
bdt = AdaBoostClassifier(DecisionTreeClassifier(max_depth=2, \
												min_samples_split=20,
												min_samples_leaf=5),
						 algorithm='SAMME', \
						 n_estimators=200, \
						 learning_rate=0.8)
r1 = bdt.fit(X, y)
print('0' * 100)
print('Score:', bdt.score(X, y))
bdt = AdaBoostClassifier(DecisionTreeClassifier(max_depth=2,
												min_samples_split=5,
												min_samples_leaf=5), \
						 algorithm='SAMME', \
						 n_estimators=300, \
						 learning_rate=0.8
						 )
bdt.fit(X, y)
print('1' * 100)
print("Score:", bdt.score(X, y))
bdt = AdaBoostClassifier(DecisionTreeClassifier(max_depth=2, \
												min_samples_split=20, \
												min_samples_leaf=5),
						 algorithm='SAMME', \
						 n_estimators=700, \
						 learning_rate=0.7)
bdt.fit(X, y)
print('2' * 100)
print("Score:", bdt.score(X, y))
