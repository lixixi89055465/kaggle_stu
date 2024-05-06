# -*- coding: utf-8 -*-
# @Time : 2024/5/6 21:27
# @Author : nanji
# @Site :https://zhuanlan.zhihu.com/p/141506312
# @File : testSelectFromModel01.py
# @Software: PyCharm 
# @Comment : 
import warnings

warnings.filterwarnings('ignore')
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.datasets import load_iris
from sklearn.feature_selection import SelectFromModel

iris = load_iris()
X, y = iris.data, iris.target
clf = ExtraTreesClassifier()
clf = clf.fit(X, y)
model = SelectFromModel(clf, prefit=True)
X_new = model.transform(X)
print('0' * 100)
print(X_new.shape)
