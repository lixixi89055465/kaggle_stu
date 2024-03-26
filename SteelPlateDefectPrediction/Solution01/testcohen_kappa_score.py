# -*- coding: utf-8 -*-
# @Time : 2024/3/26 23:26
# @Author : nanji
# @Site :  https://juejin.cn/post/7033544060019146782
# @File : testcohen_kappa_score.py
# @Software: PyCharm 
# @Comment : 
from sklearn.metrics import cohen_kappa_score, confusion_matrix

y_true = [2, 0, 2, 2, 0, 1]
y_pred = [0, 0, 2, 2, 0, 2]
print(confusion_matrix(y_true, y_pred))
print("-----------")
print(cohen_kappa_score(y_true, y_pred))
