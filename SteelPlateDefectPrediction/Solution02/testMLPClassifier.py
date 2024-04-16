# -*- coding: utf-8 -*-
# @Time : 2024/4/16 11:15
# @Author : nanji
# @Site : https://www.jianshu.com/p/71fde5d90136
# @File : testMLPClassifier.py
# @Software: PyCharm 
# @Comment :
from sklearn.neural_network import MLPClassifier
import gzip
import pickle
from sklearn.neural_network import MLPClassifier

X = [[0., 0.], [1., 1.]]
y = [0, 1]
clf = MLPClassifier(solver='lbfgs', \
					alpha=1e-5, \
					hidden_layer_sizes=(5, 2), \
					random_state=1)
r1 = clf.fit(X, y)
print(r1)
print('0'*100)
print(clf.predict([[2, 2], [-1, -2]]))
print('1'*100)
print(clf.predict_proba([[2, 2], [-1, -2]]))
