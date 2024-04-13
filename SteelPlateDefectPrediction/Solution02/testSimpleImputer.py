# -*- coding: utf-8 -*-
# @Time    : 2024/4/13 上午8:57
# @Author  : nanji
# @Site    : https://blog.csdn.net/qq_43965708/article/details/115625768
# @File    : testSimpleImputer.py
# @Software: PyCharm 
# @Comment :
from sklearn.impute import SimpleImputer
import numpy as np

X = np.array([[1, 2, 3],
			  [4, 5, 6],
			  [7, 8, 9]])
X1 = np.array([[1, 2, np.nan],
			   [4, np.nan, 6],
			   [np.nan, 8, 9]])
imp=SimpleImputer(missing_values=np.nan,strategy='mean')
imp.fit(X)
print(imp.transform(X1))
print('0'*100)
X1 = np.array([[1, 2, np.nan],
			   [4, np.nan, 6],
			   [np.nan, 8, 9]])

imp=SimpleImputer(missing_values=np.nan,strategy='mean')
print(imp.fit_transform(X1))
print('1'*100)
print(imp.get_params())
print('2'*100)

X = np.array([[1, 2, 3],
             [4, 5, 6],
             [7, 8, 9]])

imp = SimpleImputer(missing_values=1, strategy='constant', fill_value=666)
print(imp.fit_transform(X))
print('3'*100)
X1 = np.array([[1, 2, np.nan],
               [4, np.nan, 6],
               [np.nan, 8, 9]])
print('4'*100)

imp = SimpleImputer(missing_values=np.nan, strategy='mean', add_indicator=True)
X1 = imp.fit_transform(X1)
print(X1)
print(imp.inverse_transform(X1))
print('5'*100)

X = np.array([[1, 2, 3],
             [4, 5, 6],
             [7, 8, 9]])

imp = SimpleImputer(missing_values=1, strategy='constant', fill_value=None)
print(imp.fit_transform(X))

