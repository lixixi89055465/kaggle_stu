# -*- coding: utf-8 -*-
# @Time : 2024/3/25 22:51
# @Author : nanji
# @Site : https://scikit-learn.org.cn/view/643.html
# @File : testRepeatedStratifiedKFold.py
# @Software: PyCharm 
# @Comment : 10.11
import numpy as np
from sklearn.model_selection import RepeatedStratifiedKFold

X = np.array([[1, 2], [3, 4], [1, 2], [3, 4]])
y = np.array([0, 0, 1, 1])
rskf = RepeatedStratifiedKFold(n_splits=2, n_repeats=2,
							   random_state=36851234)
for train_index, test_index in rskf.split(X, y):
	print("TRAIN:", train_index, "TEST:", test_index)
	X_train, X_test = X[train_index], X[test_index]
	y_train, y_test = y[train_index], y[test_index]
