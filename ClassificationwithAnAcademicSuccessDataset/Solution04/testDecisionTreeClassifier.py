# -*- coding: utf-8 -*-
# @Time : 2024/11/2 20:46
# @Author : nanji
# @Site : https://scikit-learn.cn/stable/modules/tree.html
# @File : testDecisionTreeClassifier.py
# @Software: PyCharm 
# @Comment :
from sklearn.tree import DecisionTreeClassifier
import numpy as np

X = np.array([np.nan, -1, np.nan, 1]).reshape(-1, 1)
y = [0, 0, 1, 1]

tree = DecisionTreeClassifier(random_state=0).fit(X, y)

X_test = np.array([np.nan]).reshape(-1, 1)
res = tree.predict(X_test)
print(res)
