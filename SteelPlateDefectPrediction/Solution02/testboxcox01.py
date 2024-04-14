# -*- coding: utf-8 -*-
# @Time : 2024/4/14 16:08
# @Author : nanji
# @Site : https://blog.csdn.net/BF02jgtRS00XKtCx/article/details/108612854
# @File : testboxcox01.py.py
# @Software: PyCharm 
# @Comment : 
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=[12,8])
data1 = stats.lomax(c=14).rvs(size=500) #生成帕累托分布数据，数据每次生成时可能不同
sns.distplot(data1)
# plt.show()

fig, ax = plt.subplots(figsize=[12,8])
converted_data1 = stats.boxcox(data1)[0] #对数据进行BOX-COX变换
sns.displot(converted_data1)

fig, ax = plt.subplots(figsize=[12,8])
prob = stats.probplot(converted_data1, dist=stats.norm, plot=ax) #生成Q-Q图
plt.show()