# -*- coding: utf-8 -*-
# @Time : 2024/10/29 22:03
# @Author : nanji
# @Site : https://zhuanlan.zhihu.com/p/276601891
# https://www.jianshu.com/p/e1bda6355452
#
# @File : testhyperopt.py
# @Software: PyCharm 
# @Comment :
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from hyperopt import tpe, hp, fmin, STATUS_OK, Trials
from hyperopt.pyll.base import scope

import warnings

warnings.filterwarnings('ignore')

data = pd.read_csv("../input/archive/train.csv")
print(data.head())
print(data.shape)
print(list(data.columns))

X = data.drop("price_range", axis=1).values
y = data.price_range.values
# 标准化特征变量

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print('0' * 100)

print(X_scaled.shape)
space = {
    "n_estimators": hp.choice("n_estimators", [100, 200, 300, 400, 500, 600]),
    # "max_depth": hp.quniform("max_depth", 1, 16, 1),
    "criterion": hp.choice("criterion", ["gini", "entropy"]),
}


def hyperparameter_tuning(params):
    clf = RandomForestClassifier(**params, n_jobs=-1)
    acc = cross_val_score(clf, X_scaled, y, scoring="accuracy").mean()
    return {"loss": -acc, "status": STATUS_OK}


# 初始化Trial 对象
trials = Trials()

best = fmin(
    fn=hyperparameter_tuning,
    space=space,
    algo=tpe.suggest,
    max_evals=100,
    trials=trials
)

print("Best: {}".format(best))
print('2' * 100)
print(trials.results)
