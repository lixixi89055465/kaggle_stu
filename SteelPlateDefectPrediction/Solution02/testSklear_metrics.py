# -*- coding: utf-8 -*-
# @Time : 2024/5/8 13:30
# @Author : nanji
# @Site : https://blog.csdn.net/m0_37876503/article/details/107823115
# @File : testSklear_metrics.py
# @Software: PyCharm 
# @Comment :

from sklearn.metrics import max_error

# y_true = [3, 2, 7, 1]
# y_pred = [9, 2, 7, 1]
# r1 = max_error(y_true, y_pred)
# print(r1)

from sklearn.metrics import mean_absolute_error

# y_true = [3, -0.5, 2, 7]
# y_pred = [2.5, 0.0, 2, 8]
# r2 = mean_absolute_error(y_true, y_pred)
# print(r2)
# print('0' * 100)
#
# y_true = [[0.5, 1], [-1, 1], [7, -6]]
# y_pred = [[0, 2], [-1, 2], [8, -5]]
# r3 = mean_absolute_error(y_true, y_pred, multioutput='raw_values')
# print('1' * 100)
# print(r3)
#
# r4 = mean_absolute_error(y_true, y_pred, multioutput=[0.3, 0.7])
# print('2'*100)
# print(r4)
from sklearn.metrics import mean_squared_error

# y_true = [3, -0.5, 2, 7]
# y_pred = [2.5, 0.0, 2, 8]

# r5 = mean_squared_error(y_true, y_pred)
# print('3' * 100)
# print(r5)
from sklearn.metrics import mean_squared_log_error

y_true = [3, 5, 2.5, 7]
y_pred = [2.5, 5, 4, 8]
r6 = mean_squared_log_error(y_true, y_pred)
print(r6)
