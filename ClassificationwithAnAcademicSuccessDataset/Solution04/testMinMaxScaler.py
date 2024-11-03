# -*- coding: utf-8 -*-
# @Time : 2024/11/3 15:00
# @Author : nanji
# @Site : 
# @File : testMinMaxScaler.py
# @Software: PyCharm 
# @Comment :
from sklearn.preprocessing import MinMaxScaler

data = [[-1, 2], [-0.5, 6], [0, 10], [1, 18]]

# 不太熟悉numpy的小伙伴，能够判断data的结构吗？
# 如果换成表是什么样子？
import pandas as pd

pd.DataFrame(data)
# 实现归一化
scaler = MinMaxScaler()  # 实例化
scaler = scaler.fit(data)  # fit，在这里本质是生成min(x)和max(x)
result = scaler.transform(data)  # 通过接口导出结果
print(result)
result_ = scaler.fit_transform(data)  # 训练和导出结果一步达成
scaler.inverse_transform(result)  # 将归一化后的结果逆转
data = [[-1, 2], [-0.5, 6], [0, 10], [1, 18]]
scaler = MinMaxScaler(feature_range=[5, 10])  # 依然实例化
result = scaler.fit_transform(data)  # fit_transform一步导出结果
print(result)
print('3' * 100)

import numpy as np

X = np.array([[-1, 2], [-0.5, 6], [0, 10], [1, 18]])
# 归一化
X_nor = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
print(X_nor)
# 逆转归一化
X_returned = X_nor * (X.max(axis=0) - X.min(axis=0)) + X.min(axis=0)
print(X_returned)

print('2' * 100)
from sklearn.preprocessing import StandardScaler

data = [[-1, 2], [-0.5, 6], [0, 10], [1, 18]]
scaler = StandardScaler()  # 实例化
scaler.fit(data)  # fit，本质是生成均值和方差
# scaler.mean_                            #查看均值的属性mean_
print('1' * 100)
print(scaler.mean_)
print(scaler.var_)
x_std = scaler.transform(data)  # 通过接口导出结果
print('2' * 100)
print(x_std.mean())
print(x_std.std())
print('3' * 100)
