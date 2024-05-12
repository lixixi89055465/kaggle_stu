# -*- coding: utf-8 -*-
# @Time    : 2024/5/12 下午4:45
# @Author  : nanji
# @Site    : https://blog.csdn.net/TeFuirnever/article/details/100276569
# @File    : testAdaBoostClassifier01.py
# @Software: PyCharm 
# @Comment :
from sklearn.ensemble import AdaBoostClassifier
from sklearn.datasets import make_classification

X, y = make_classification(n_samples=1000, \
						   n_features=4, \
						   n_informative=2, \
						   n_redundant=0,
						   random_state=0, shuffle=False)
clf = AdaBoostClassifier(n_estimators=100, random_state=0)
clf.fit(X, y)
print('0' * 100)
print(clf.feature_importances_)
clf.predict([[0, 0, 0, 0]])
r1 = clf.score(X, y)
print(r1)
