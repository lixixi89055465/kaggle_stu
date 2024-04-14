# -*- coding: utf-8 -*-
# @Time : 2024/4/14 15:57
# @Author : nanji
# @Site : https://blog.csdn.net/qq_42774234/article/details/130059235
# @File : testboxcox.py
# @Software: PyCharm 
# @Comment : 
from scipy import stats

# 假设有一组数据x
x = [1, 2, 3, 4, 5]

# 进行Box-Cox变换 convert_res是输出结果
convert_res, _ = stats.boxcox(x)

print(convert_res)

# from scipy.special import inv_boxcox
# x_inv = inv_boxcox(convert_res, _)
#
# print(x_inv)