# -*- coding: utf-8 -*-
# @Time    : 2024/5/12 下午6:12
# @Author  : nanji
# @Site    :https://zhuanlan.zhihu.com/p/607273766
# @File    : testSMOTEENN01.py
# @Software: PyCharm 
# @Comment :
from imblearn.combine import SMOTEENN
from sklearn.datasets import make_classification

X, y = make_classification(n_samples=10000, n_features=2, \
						   n_informative=2, n_classes=3,
						   n_redundant=0, n_repeated=0, \
						   n_clusters_per_class=1, \
						   weights=[0.01, 0.05, 0.94], \
						   class_sep=0.8, \
						   random_state=0)
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
# bc = BaggingClassifier(base_estimator=DecisionTreeClassifier(), \
# 					   random_state=0)
# bc.fit(X_train, y_train)
# y_pred = bc.predict(X_test)
# r1 = balanced_accuracy_score(y_test, y_pred)
# print(r1)
from imblearn.ensemble import BalancedBaggingClassifier

# bbc = BalancedBaggingClassifier( \
# 	base_estimator=DecisionTreeClassifier(), \
# 	sampling_strategy='auto', \
# 	replacement=False, \
# 	random_state=0)
# bbc.fit(X_train, y_train)
# y_pred = bbc.predict(X_test)
# r2 = balanced_accuracy_score(y_test, y_pred)
# print(r2)

from imblearn.ensemble import BalancedRandomForestClassifier

# brf = BalancedRandomForestClassifier(n_estimators=100, random_state=0)
# brf.fit(X_train, y_train)
# y_pred = brf.predict(X_test)
# r3 = balanced_accuracy_score(y_test, y_pred)
# print(r3)
from imblearn.ensemble import RUSBoostClassifier

rusboost = RUSBoostClassifier( \
	n_estimators=200, \
	algorithm='SAMME.R', \
	random_state=0)
rusboost.fit(X_train, y_train)
y_pred = rusboost.predict(X_test)
r4 = balanced_accuracy_score(y_test, y_pred)
print(r4)

# from imblearn.ensemble import EasyEnsembleClassifier
#
# eec = EasyEnsembleClassifier(random_state=0)
# eec.fit(X_train, y_train)
# y_pred = eec.predict(X_test)
# r5 = balanced_accuracy_score(y_test, y_pred)
# print(r5)
