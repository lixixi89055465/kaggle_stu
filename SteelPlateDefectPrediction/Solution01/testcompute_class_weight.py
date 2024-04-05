# -*- coding: utf-8 -*-
# @Time : 2024/4/4 15:00
# @Author : nanji
# @Site : https://blog.csdn.net/Ving_x/article/details/134059550
# @File : testcompute_class_weight.py
# @Software: PyCharm 
# @Comment : 
from sklearn.utils.class_weight import compute_class_weight

classes = [0, 1, 2]
y = [0, 1, 1, 2, 2, 2]  # 假设训练数据的标签

weights = compute_class_weight({0: 1, 1: 2, 2: 3}, classes, y)
print(weights)  # 输出: [1.5, 1.0, 0.66666667]
