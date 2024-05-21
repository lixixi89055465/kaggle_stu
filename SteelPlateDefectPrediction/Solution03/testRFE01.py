# -*- coding: utf-8 -*-
# @Time : 2024/5/21 22:33
# @Author : nanji
# @Site :https://zhuanlan.zhihu.com/p/64900887
# @File : testRFE01.py
# @Software: PyCharm 
# @Comment :
from sklearn.feature_selection import RFE,RFECV
from sklearn.svm import LinearSVC
from sklearn.datasets import load_iris
from sklearn import model_selection

iris=load_iris()
X,y=iris.data,iris.target
## 特征提取
estimator=LinearSVC()
selector=RFE(estimator=estimator,n_features_to_select=2)

X_t=selector.fit_transform(X,y)
# 切分测试集与验证机

