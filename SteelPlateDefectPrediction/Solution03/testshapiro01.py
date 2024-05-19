# -*- coding: utf-8 -*-
# @Time : 2024/5/18 16:22
# @Author : nanji
# @Site :https://blog.csdn.net/qq_20207459/article/details/102596780
# @File : testshapiro01.py
# @Software: PyCharm 
# @Comment :
from scipy import stats
import numpy as np

# 创建一个示例数据集（这里使用正态分布数据）
data = np.random.normal(0, 1, 100)

# 执行Shapiro-Wilk正态性检验
statistic, p_value = stats.shapiro(data)

# 输出检验结果
print("Shapiro-Wilk统计量:", statistic)
print("p-value:", p_value)

# 根据p-value做出决策
alpha = 0.05  # 显著性水平
if p_value > alpha:
	print("不能拒绝零假设，数据可能服从正态分布")
else:
	print("拒绝零假设，数据不服从正态分布")