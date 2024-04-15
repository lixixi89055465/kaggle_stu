# -*- coding: utf-8 -*-
# @Time : 2024/4/15 14:19
# @Author : nanji
# @Site : https://blog.csdn.net/fengdu78/article/details/132843843
# @File : testHistGradientBoostingClassifier01.py
# @Software: PyCharm 
# @Comment : 
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.datasets import make_hastie_10_2

X, y = make_hastie_10_2(random_state=0)
X_train, X_test = X[:2000], X[2000:]
y_train, y_test = y[:2000], y[2000:]
clf = HistGradientBoostingClassifier(max_iter=100).fit(X_train, y_train)
print(clf.score(X_test, y_test))


