# -*- coding: utf-8 -*-
# @Time : 2024/5/31 19:36
# @Author : nanji
# @Site : https://www.cnblogs.com/wang_yb/p/17997343
# @File : testbrier_score_loss02.py
# @Software: PyCharm 
# @Comment :
from sklearn.metrics import brier_score_loss
import numpy as np

n = 100
y_true = np.random.randint(0, 2, n)
y_prob = np.random.rand(n)

print(y_true.min())
print(y_true.max())
s = brier_score_loss(y_true, y_prob)
print("brier score loss：{}".format(s))
print('1'*100)
print(y_prob[:10])

