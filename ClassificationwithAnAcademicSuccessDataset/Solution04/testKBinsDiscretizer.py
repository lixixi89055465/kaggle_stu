# -*- coding: utf-8 -*-
# @Time : 2024/11/3 14:48
# @Author : nanji
# @Site : 
# @File : testKBinsDiscretizer.py
# @Software: PyCharm 
# @Comment :

import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import KBinsDiscretizer

X = np.array([
    [-3., 5., 15],
    [0., 6., 14],
    [6., 3., 11]
])
est = preprocessing.KBinsDiscretizer(n_bins=[3, 2, 2],
                                     encode='ordinal').fit(X)
res = est.transform(X)
print(res)
print('0' * 100)
X = [[-2, 1, -4, -1],
     [-1, 2, -3, -0.5],
     [0, 3, -2, 0.5],
     [1, 4, -1, 2]]
est = KBinsDiscretizer(n_bins=3, encode='ordinal', strategy='uniform')
res = est.fit(X)
Xt = est.transform(X)
print('2' * 100)
print(Xt)

print('3' * 100)
print(est.bin_edges_[0])
res = est.inverse_transform(Xt)
print('4' * 100)
print(res)
