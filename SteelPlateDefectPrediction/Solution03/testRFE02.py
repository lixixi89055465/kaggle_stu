# -*- coding: utf-8 -*-
# @Time : 2024/5/21 23:09
# @Author : nanji
# @Site :https://zhuanlan.zhihu.com/p/618646313
# @File : testRFE02.py
# @Software: PyCharm 
# @Comment :
from sklearn.datasets import load_iris
from sklearn.feature_selection import RFE
from sklearn.tree import DecisionTreeClassifier

# data = load_iris()
# X, y = data.data, data.target
# rfe = RFE(estimator=DecisionTreeClassifier(), n_features_to_select=2)
# rfe.fit(X, y)
# print('Selected Features:')
# print('data.feature_names:')
# print(data.feature_names)
# for i in range(len(data.feature_names)):
# 	if rfe.support_[i]:
# 		print(data.feature_names[i])


from sklearn.datasets import load_boston
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression

data = load_boston()
X, y = data.data, data.target
## Create RFE object
rfe = RFE(estimator=LinearRegression(), n_features_to_select=5)
## Fit RFE
rfe.fit(X, y)
## Print selected features
print('Selected Features:')
for i in range(len(data.feature_names)):
    if rfe.support_[i]:
        print(data.feature_names[i])