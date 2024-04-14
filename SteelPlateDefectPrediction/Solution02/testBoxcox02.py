# -*- coding: utf-8 -*-
# @Time : 2024/4/14 16:32
# @Author : nanji
# @Site : https://blog.csdn.net/BF02jgtRS00XKtCx/article/details/108612854
# @File : testBoxcox02.py
# @Software: PyCharm 
# @Comment :
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
import matplotlib.pyplot as plt

state = np.random.RandomState(20)  # 设置随机状态
data2 = state.exponential(size=500)  # 生成指数分布数据集
converted_data2 = stats.boxcox(data2)[0]  # 将数据进行BOX-COX变换

fig, ax = plt.subplots(figsize=[12,8])
sns.distplot(data2) #绘制原始指数分布数据的直方图
plt.show()

fig, ax = plt.subplots(figsize=[12, 8])
sns.distplot(converted_data2)  # 绘制转换后数据的直方图
plt.show()