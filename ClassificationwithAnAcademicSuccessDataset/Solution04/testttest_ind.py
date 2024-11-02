# -*- coding: utf-8 -*-
# @Time : 2024/11/2 21:07
# @Author : nanji
# @Site : https://blog.csdn.net/qq_37006625/article/details/127937428
# @File : testttest_ind.py
# @Software: PyCharm 
# @Comment :
from scipy import stats

A = stats.norm.rvs(loc=1, scale=1, size=(100))  # 生成A公司的销售额
B = stats.norm.rvs(loc=3, scale=1, size=(100))  # 生成B公司的销售额
res = stats.levene(A, B)  # 进行levene检验
print(res)
C = stats.ttest_ind(A, B, equal_var=True)
print(C)
