# -*- coding: utf-8 -*-
# @Time : 2024/5/1 17:18
# @Author : nanji
# @Site : https://zhuanlan.zhihu.com/p/141506312
# @File : testSelectKBest01.py
# @Software: PyCharm 
# @Comment :
from sklearn.datasets import load_iris
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

iris = load_iris()
X, y = iris.data, iris.target
X_new = SelectKBest(chi2, k=2).fit_transform(X, y)
print(X_new)
