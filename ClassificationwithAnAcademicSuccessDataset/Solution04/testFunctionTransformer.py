# -*- coding: utf-8 -*-
# @Time : 2024/11/3 13:55
# @Author : nanji
# @Site : https://www.cnblogs.com/qiu-hua/p/14903451.html
# @File : testFunctionTransformer.py
# @Software: PyCharm 
# @Comment :

# import numpy as np
# from sklearn.preprocessing import FunctionTransformer
#
# transformer = FunctionTransformer(np.log1p, validate=True)
# X = np.array([[0, 1], [2, 3]])
# res = transformer.transform(X)
# print(res)


import numpy as np
from sklearn.preprocessing import FunctionTransformer

transformer = FunctionTransformer(np.log1p)
X = np.array([[0, 1], [2, 3]])
res = transformer.transform(X)
print(res)
