# -*- coding: utf-8 -*-
# @Time : 2024/11/3 0:47
# @Author : nanji
# @Site :https://zhuanlan.zhihu.com/p/366787872
# @File : testBernoulliNB02.py
# @Software: PyCharm 
# @Comment :
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB, ComplementNB
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import cross_val_score

data = load_breast_cancer()
X, y = data.data, data.target
nb1 = GaussianNB()
nb2 = MultinomialNB()
nb3 = BernoulliNB()
nb4 = ComplementNB()
# for model in [nb1, nb2, nb3, nb4]:
#     scores = cross_val_score(model, X, y, cv=10,
#                              scoring='accuracy')
#     print('Accuracy:{:.4f}'.format(scores.mean()))

from sklearn import preprocessing

enc = preprocessing.OneHotEncoder()
from sklearn.datasets import load_iris

data = load_iris()
X, y = data.data, data.target
# enc.fit(X)
# array = enc.transform(X).toarray()
# print('array.shape')
# print(array.shape)

from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB, ComplementNB, CategoricalNB
from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score

data = load_iris()
X, y = data.data, data.target
enc.fit(X)
array = enc.transform(X).toarray()

gn1 = GaussianNB()
gn2 = MultinomialNB()
gn3 = BernoulliNB()
gn4 = ComplementNB()

for model in [gn1, gn2, gn3, gn4]:
    scores = cross_val_score(model, array, y, cv=10, scoring='accuracy')
    print('Accuracy:{:.4f}'.format(scores.mean()))
