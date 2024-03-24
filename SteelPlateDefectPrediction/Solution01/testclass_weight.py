# -*- coding: utf-8 -*-
# @Time : 2024/3/24 20:59
# @Author : nanji
# @Site : https://blog.csdn.net/Ving_x/article/details/134059550
# @File : testclass_weight.py
# @Software: PyCharm 
# @Comment : 
from sklearn.utils.class_weight import compute_class_weight

class_weight = 'balanced'
classes = [0, 1, 2]
y = [0, 1, 1, 2, 2, 2]  # 假设训练数据的标签

weights = compute_class_weight(class_weight, classes, y)
print(weights)  # 输出: [1.5, 1.0, 0.66666667]

# coding:utf-8

# from sklearn.utils.class_weight import compute_class_weight
# class_weight = 'balanced'
# label = [0] * 9 + [1] * 1 + [2, 2]
# print(label)  # [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 2]
# classes = [0, 1, 2]
# weight = compute_class_weight(class_weight, classes, label)
# print(weight)  # [ 0.44444444 4.         2.        ]
# print(.44444444 * 9)  # 3.99999996
# print(4 * 1)  # 4
# print(2 * 2)  # 4
