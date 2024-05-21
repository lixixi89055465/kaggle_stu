# -*- coding: utf-8 -*-
# @Time : 2024/5/21 18:01
# @Author : nanji
# @Site : https://blog.csdn.net/iizhuzhu/article/details/105031532
# @File : testSelectKBest.py
# @Software: PyCharm 
# @Comment :
from sklearn.datasets import load_iris
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

X, y = load_iris(return_X_y=True)
print(X.shape)

X_new = SelectKBest(chi2, k=2).fit_transform(X, y)
print('0' * 100)
print(X_new.shape)
