# -*- coding: utf-8 -*-
# @Time : 2024/5/7 15:37
# @Author : nanji
# @Site :https://vimsky.com/examples/usage/python-sklearn.preprocessing.FunctionTransformer-sk.html
# @File : testFunctionTransformer01.py
# @Software: PyCharm 
# @Comment : 

import numpy as np
from sklearn.preprocessing import FunctionTransformer

transformer = FunctionTransformer(np.log1p)
X = np.array([[0, 1], [2, 3]])
r1 = transformer.transform(X)
print('0' * 100)
print(r1.shape)
print(r1)
