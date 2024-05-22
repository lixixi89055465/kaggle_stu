# -*- coding: utf-8 -*-
# @Time : 2024/5/22 21:25
# @Author : nanji
# @Site : https://vimsky.com/examples/usage/python-sklearn.metrics.make_scorer-sk.html
# @File : testmake_scorer01.py
# @Software: PyCharm 
# @Comment :

from sklearn.metrics import fbeta_score, make_scorer

ftwo_scores = make_scorer(fbeta_score, beta=2)
print('0' * 100)
print(ftwo_scores)
from sklearn.model_selection import GridSearchCV
from sklearn.svm import LinearSVC

grid = GridSearchCV(LinearSVC(), param_grid={'C': [1, 10]},
					scoring=ftwo_scores)

# model = LGBMRegressor(max_depth=5,num_leaves=10,objective="regression")
# score_ = cross_val_score(model,X = X_train,y=Y_train,verbose=0,
# 						 scoring=make_scorer(mean_squared_error))
# print(score_)
