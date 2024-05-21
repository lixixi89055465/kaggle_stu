# -*- coding: utf-8 -*-
# @Time : 2024/5/21 17:44
# @Author : nanji
# @Site : https://blog.csdn.net/everyxing1007/article/details/126032477
# @File : testmutual_info_classif.py
# @Software: PyCharm 
# @Comment :

import pandas as pd
import numpy as np
names = ['label', 'L1', 'L2', 'L3', 'L4', 'L5', 'L6', 'L7', 'L8', 'L9', 'L10', 'L11', 'L12']
dataframe = pd.read_csv("./L_roudoukou_data_LSTM_CNN.csv", names=names)  # 读取数据集
print(dataframe.head())
array = dataframe.values
# **注：不要选到标签栏，否则XY就会包含字符串类型，后面无法计算。**
X = array[1:, 1:13]  # 选取前1~13列为特征变量，就是要筛选的特征。
Y = array[1:, 0]  # 选取label为目标变量，就是标签。
# print(X[0:5, :])    # 打印前5行特征
# print(Y[0:5])    # 打印前5行标签

print('0'*100)
# from sklearn.feature_selection import mutual_info_classif
# import matplotlib.pyplot as plt
# importances = mutual_info_classif(X, Y)
# feat_importances = pd.Series(importances, df.columns[0:len(df.columns)-1])
# feat_importances.plot(kind='barh', color='teal')
# plt.savefig('./mutual_info_classif.png')
# plt.show()
# plt.close()
