# -*- coding: utf-8 -*
from sklearn import datasets, linear_model
from sklearn.model_selection import cross_val_score

diabetes = datasets.load_diabetes()
X = diabetes.data[:150]
y = diabetes.target[:150]
print(X.shape)
print(y.shape)
print('1'*100)
lasso = linear_model.Lasso()
print(cross_val_score(lasso, X, y, cv=3))
