# -*- coding: utf-8 -*-
# @Time : 2024/4/15 14:51
# @Author : nanji
# @Site : https://blog.csdn.net/fengdu78/article/details/132843843
# @File : testHistGradientBoostingClassifier02.py
# @Software: PyCharm 
# @Comment :
from sklearn.ensemble import HistGradientBoostingClassifier
import numpy as np

X = np.array([0, 1, 2, np.nan]).reshape(-1, 1)
y = [0, 0, 1, 1]
gbdt = HistGradientBoostingClassifier(min_samples_leaf=1).fit(X, y)
print(gbdt.predict(X))

print('0' * 100)
X = np.array([0, np.nan, 1, 2, np.nan]).reshape(-1, 1)
y=[0,1,0,0,1]


