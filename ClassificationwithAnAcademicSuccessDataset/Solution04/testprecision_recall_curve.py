# -*- coding: utf-8 -*-
# @Time : 2024/11/2 16:00
# @Author : nanji
# @Site : https://blog.csdn.net/weixin_44012667/article/details/139988782
# @File : testprecision_recall_curve.py
# @Software: PyCharm 
# @Comment :

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

# 生成一个二分类数据集
X, y = make_classification(n_samples=1000, n_features=20, random_state=42)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 训练一个逻辑回归模型
model = LogisticRegression()
model.fit(X_train, y_train)

# 预测测试集的概率
y_prob = model.predict_proba(X_test)
y_prob = y_prob[:, 1]

# 计算精确率和召回率
precision, recall, _ = precision_recall_curve(y_test, y_prob)

# 计算AP
ap = average_precision_score(y_test, y_prob)
print(f"AP: {ap:.2f}")

# 绘制PR曲线
plt.plot(recall, precision, marker='.', label='Logistic Regression (AP = {:.2f})'.format(ap))
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend()
plt.show()
