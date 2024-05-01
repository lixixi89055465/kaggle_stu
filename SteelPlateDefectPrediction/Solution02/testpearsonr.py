# -*- coding: utf-8 -*-
# @Time : 2024/5/1 18:19
# @Author : nanji
# @Site :https://zhuanlan.zhihu.com/p/141506312
# @File : testpearsonr.py
# @Software: PyCharm 
# @Comment :
import numpy as np
from scipy.stats import pearsonr

np.random.seed(0)
size = 300
x = np.random.normal(0, 1, size)
y = x + np.random.normal(0, 1, size)

# 输入：特征矩阵和目标向量
# 输出： 相关系数值和p值
# print('Pearson 相关系数:', pearsonr(x, y))
print('0' * 100)
x = np.random.uniform(-1, 1, 100)
print(pearsonr(x, x ** 2)[0])
