# -*- coding: utf-8 -*-
# @Time : 2024/6/6 14:20
# @Author : nanji
# @Site : 
# @File : test01.py
# @Software: PyCharm 
# @Comment : 
import warnings

warnings.filterwarnings('ignore')
import numpy as np

# 示例数据
data = np.array([1, 2, 2, 3, 3, 3, 4, 4, 4, 4])
weights = np.array([100, 1, 2, 10, 1, 2, 5, 1, 2, 1])

# 计算每个元素的累计权重
cumulative_weights = np.add.accumulate(weights)

# 找到众数的索引
majority_index = np.argmax(np.bincount(data, cumulative_weights))

# 找到众数
majority_element = data[majority_index]

print("众数:", majority_element)