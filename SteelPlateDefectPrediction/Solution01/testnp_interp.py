# -*- coding: utf-8 -*-
# @Time : 2024/4/3 23:37
# @Author : nanji
# @Site : https://blog.csdn.net/qq_41813454/article/details/134803345
# @File : testnp_interp.py
# @Software: PyCharm 
# @Comment :
import numpy as np

# # 已知数据点
# xp = np.array([0, 1, 2, 3])
# # fp = np.array([0, 1, 4, 9])
# fp = np.array([0, 2, 4, 6])
#
# # 需要进行插值的点
# x = 1.5
#
# # 使用np.interp进行插值
# result = np.interp(x, xp, fp)
# print(result)  # 输出：2.5


# 已知数据点（周期性数据）
xp = np.array([0, 1, 2, 3])
fp = np.array([0, 1, 4, 9])
period = 4  # 数据周期为4

# 需要进行插值的点（超出已知数据点范围）
x = 5  # 实际等效于x=1（因为5 mod 4 = 1）

# 使用np.interp进行插值，并指定period参数
result = np.interp(x, xp, fp, period=period)
print(result)  # 输出：1（因为x等效于1，所以返回fp[1]的值）
