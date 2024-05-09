# -*- coding: utf-8 -*-
# @Time : 2024/4/16 10:39
# @Author : nanji
# @Site : https://blog.csdn.net/tMb8Z9Vdm66wH68VX1/article/details/130177587
# @File : testKNNImputer.py
# @Software: PyCharm 
# @Comment :
import numpy as np
from sklearn.metrics import nan_euclidean_distances
from sklearn.impute import KNNImputer

X = [
	[3, np.nan, 5],
	[1, 0, 0],
	[3, 3, 3]
]

# r1=nan_euclidean_distances(X,X)
# print(r1)

# imputer = KNNImputer(n_neighbors=2)
# r2=imputer.fit_transform(X)
# print('3'*100)
# print(r2)

import pandas as pd
data = [
	['tom', 10, 'male'],
	['nicky', 15, 'female'],
	['juli', 14, None],
	['feni', 12, 'female'],
	['johnny', 9, 'male'],
]
df=pd.DataFrame(data,columns=['Name','Age','Gender'])
print(df)

print('0'*100)
from sklearn.impute import KNNImputer
imputer=KNNImputer(n_neighbors=2)
df_filled=imputer.fit_transform(df[['Age','Gender']])
print(df_filled)