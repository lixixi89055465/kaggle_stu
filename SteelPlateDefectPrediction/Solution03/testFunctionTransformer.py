# -*- coding: utf-8 -*-
# @Time : 2024/5/21 13:43
# @Author : nanji
# @Site :https://scikit-learn.org.cn/view/725.html
# @File : testFunctionTransformer.py
# @Software: PyCharm 
# @Comment : 

import numpy as np
from sklearn.preprocessing import FunctionTransformer

# transformer = FunctionTransformer(np.log1p)
# X = np.array([[0, 1], [2, 3]])
# r1 = transformer.transform(X)
# print(r1)
print('1' * 100)
# 创建矩阵
features = np.array([[2, 3],
					 [2, 3],
					 [2, 3]])


# # 定义函数
def add_ten(x):
	return x + 10


# # 创建转换器
ten_transformer = FunctionTransformer(add_ten)
# 装换特征矩阵
r1 = ten_transformer.transform(features)
print(r1)

# 加载库
import pandas as pd

df = pd.DataFrame(features, columns=['feature_1', 'feature_2'])
r2 = df.apply(add_ten)
print('2'*100)
print(r2)
