# -*- coding: utf-8 -*-
# @Time : 2024/5/1 22:22
# @Author : nanji
# @Site : 
# @File : testSelectFromModel.py
# @Software: PyCharm 
# @Comment : 
import warnings

warnings.filterwarnings('ignore')
from sklearn.svm import LinearSVC
from sklearn.datasets import load_iris
from sklearn.feature_selection import SelectFromModel
import numpy as np

iris = load_iris()
X, y = iris.data, iris.target
lsvc = LinearSVC(C=0.01, penalty='l1', dual=False).fit(X, y)
model = SelectFromModel(lsvc, prefit=True)
X_new = model.transform(X)
print('0' * 100)
print(X_new.shape)
