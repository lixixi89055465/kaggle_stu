# -*- coding: utf-8 -*-
# @Time : 2024/5/31 19:23
# @Author : nanji
# @Site : https://vimsky.com/examples/usage/python-sklearn.metrics.brier_score_loss-sk.html
# @File : testbrier_score_loss01.py
# @Software: PyCharm 
# @Comment :

import numpy as np

from sklearn.metrics import brier_score_loss

y_true = np.array([0, 1, 1, 0])
y_true_categorical = np.array(['spam', 'ham', 'ham', 'spam'])
y_prob = np.array([0.1, 0.9, 0.8, 0.3])
r1 = brier_score_loss(y_true, y_prob)
print(r1)
r2 = brier_score_loss(y_true, 1 - y_prob, pos_label=0)
print('2' * 100)
print(r2)
r3 = brier_score_loss(y_true_categorical, y_prob, pos_label='ham')
print('3' * 100)
print(r3)
r4 = brier_score_loss(y_true, np.array(y_prob) > 0.5)
print('4' * 100)
print(r4)
