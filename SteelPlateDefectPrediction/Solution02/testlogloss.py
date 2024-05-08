# -*- coding: utf-8 -*-
# @Time : 2024/5/8 14:08
# @Author : nanji
# @Site :https://blog.csdn.net/ybdesire/article/details/73695163
# @File : testlogloss.py
# @Software: PyCharm 
# @Comment : 
from sklearn.metrics import log_loss
from sklearn.preprocessing import OneHotEncoder

one_hot = OneHotEncoder(sparse=False)

y_true = one_hot.fit_transform([0, 1, 3])
y_pred = one_hot.fit_transform([1, 2, 1])
r1 = log_loss(y_true, y_pred)
print(r1)
