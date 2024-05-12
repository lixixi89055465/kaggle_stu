# -*- coding: utf-8 -*-
# @Time : 2024/4/15 14:19
# @Author : nanji
# @Site : https://blog.csdn.net/fengdu78/article/details/132843843
# @File : testHistGradientBoostingClassifier01.py
# @Software: PyCharm 
# @Comment : 
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.datasets import make_hastie_10_2
#
# X, y = make_hastie_10_2(random_state=0)
# X_train, X_test = X[:2000], X[2000:]
# y_train, y_test = y[:2000], y[2000:]
# clf = HistGradientBoostingClassifier(max_iter=100).fit(X_train, y_train)
# print(clf.score(X_test, y_test))


from sklearn.ensemble import HistGradientBoostingClassifier
import numpy as np

# X = np.array([0, 1, 2, np.nan]).reshape(-1, 1)
# y = [0, 0, 1, 1]
# gbdt = HistGradientBoostingClassifier(min_samples_leaf=1).fit(X, y)
# r1 = gbdt.predict(X)
# print(r1)

# X = np.array([0, np.nan, 1, 2, np.nan]).reshape(-1, 1)
# y = [0, 1, 0, 0, 1]
# gbdt = HistGradientBoostingClassifier(min_samples_leaf=1, \
# 									  max_depth=2, \
# 									  learning_rate=1,
# 									  max_iter=1).fit(X, y)
# r2 = gbdt.predict(X)
# print(r2)

X = [[1, 0],
	 [1, 0],
	 [1, 0],
	 [0, 1]]
y = [0, 0, 1, 0]
sample_weight = [0, 0, 1, 1]
gb = HistGradientBoostingClassifier(min_samples_leaf=1)
gb.fit(X, y, sample_weight=sample_weight)
gb.predict([[0, 1]])
# r3 = gb.predict_proba([[1, 0]])[0, 1]
r3 = gb.predict_proba([[1, 0]])
print(r3)
