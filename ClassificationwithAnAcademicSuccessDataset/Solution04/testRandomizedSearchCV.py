# -*- coding: utf-8 -*-
# @Time : 2024/11/2 18:03
# @Author : nanji
# @Site :https://zhuanlan.zhihu.com/p/56010146
# @File : testRandomizedSearchCV.py
# @Software: PyCharm 
# @Comment :
import numpy as np
from scipy.stats import randint as sp_randint
from sklearn.model_selection import RandomizedSearchCV
from sklearn.datasets import load_digits
from sklearn.ensemble import RandomForestClassifier
# 载入数据
digits = load_digits()
X, y = digits.data, digits.target

# 建立一个分类器或者回归器
clf = RandomForestClassifier(n_estimators=20)

# 给定参数搜索范围：list or distribution
param_dist = {"max_depth": [3, None],                     #给定list
              "max_features": sp_randint(1, 11),          #给定distribution
              "min_samples_split": sp_randint(2, 11),     #给定distribution
              "bootstrap": [True, False],                 #给定list
              "criterion": ["gini", "entropy"]}           #给定list
# 用RandomSearch+CV选取超参数
n_iter_search = 20
random_search = RandomizedSearchCV(clf, param_distributions=param_dist,
                                   n_iter=n_iter_search, cv=5, iid=False)
random_search.fit(X, y)