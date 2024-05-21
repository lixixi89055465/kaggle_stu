# -*- coding: utf-8 -*-
# @Time : 2024/5/21 16:02
# @Author : nanji
# @Site : https://zhuanlan.zhihu.com/p/613526479
# @File : testBaggingClassifier.py
# @Software: PyCharm 
# @Comment : 
from sklearn.svm import SVC
from sklearn.ensemble import BaggingClassifier
from sklearn.datasets import make_classification

X, y = make_classification(n_samples=100, n_features=4,
						   random_state=0, \
						   n_classes=3, \
						   n_informative=3, n_redundant=1)
m1 = BaggingClassifier(base_estimator=SVC(), n_estimators=10, random_state=0).fit(X, y)
print("预测：", m1.predict([[-0.1, -2, -1, 1]]))
print("准确率：", m1.score(X, y))
