# -*- coding: utf-8 -*-
# @Time : 2024/5/28 14:48
# @Author : nanji
# @Site : https://blog.csdn.net/qq_45807032/article/details/112974494
# @File : testBoxPlot.py
# @Software: PyCharm 
# @Comment : 
import matplotlib.pyplot as plt
import numpy as np

# 生成data的数据
# np.random.seed(100)
# data = np.random.normal(size=(1000,), loc=0, scale=1)
# plt.boxplot(data)
# plt.show()

np.random.seed(100)
data = np.random.normal(size=(1000, 4), loc=0, scale=1)
print(data.shape)
plt.boxplot(data)
plt.show()
