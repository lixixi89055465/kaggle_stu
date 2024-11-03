# -*- coding: utf-8 -*-
# @Time : 2024/11/3 13:40
# @Author : nanji
# @Site : https://blog.csdn.net/CSDNXXCQ/article/details/132610459
# @File : testPowerTransformer02.py
# @Software: PyCharm 
# @Comment :
from sklearn.preprocessing import PowerTransformer
import numpy as np

data = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]])

transformer = PowerTransformer(method='yeo-johnson')
transformed_data = transformer.fit_transform(data)

print("Original Data:\n", data)
print("Transformed Data:\n", transformed_data)

print('0' * 100)
from sklearn.preprocessing import MinMaxScaler
import numpy as np

data = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]])
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)
print('1' * 100)
print(data)
print('2' * 100)
print(scaled_data)
