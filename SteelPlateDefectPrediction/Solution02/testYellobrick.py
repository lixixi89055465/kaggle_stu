# -*- coding: utf-8 -*-
# @Time : 2024/4/7 11:27
# @Author : nanji
# @Site : https://blog.csdn.net/LuohenYJ/article/details/107575710
# @File : testYellobrick.py
# @Software: PyCharm 
# @Comment :
import yellowbrick

import pandas as pd
from yellowbrick.datasets import load_bikeshare, load_concrete, load_occupancy
from sklearn.model_selection import TimeSeriesSplit
# from sklearn.metrics import classification_report
from yellowbrick.classifier import classification_report
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt

# X, y = load_occupancy()
# print(X.columns)
# # Specify the target classes
# classes = ["unoccupied", "occupied"]
# tscv = TimeSeriesSplit()
# for train_index, test_index in tscv.split(X):
# 	X_train, X_test = X.iloc[train_index], X.iloc[test_index]
# 	y_train, y_test = y.iloc[train_index], y.iloc[test_index]
# visualizer = classification_report(
# 	GaussianNB(),  #
# 	X_train,  #
# 	y_train,  #
# 	X_test,  #
# 	y_test,  #
# 	classes=classes,  #
# 	support=True)

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split as tts
from sklearn.linear_model import LogisticRegression
from yellowbrick.classifier import ConfusionMatrix

# We'll use the handwritten digits data set from scikit-learn.
# Each feature of this dataset is an 8x8 pixel image of a handwritten number.
# Digits.data converts these 64 pixels into a single array of features
#我们将使用scikit learn中的手写数字数据集。
#该数据集的每个特征都是手写数字的8x8像素图像。
# Digits.data 将这64个像素转换为一个维度数组
digits = load_digits()
X = digits.data
y = digits.target

X_train, X_test, y_train, y_test = tts(X, y, test_size =0.2, random_state=11)
X_test.shape,y_test.shape
model = LogisticRegression(multi_class="auto", solver="liblinear")

# The ConfusionMatrix visualizer taxes a model
# 混淆矩阵分类号
cm = ConfusionMatrix(model, classes=[0,1,2,3,4,5,6,7,8,9])

# Fit fits the passed model. This is unnecessary if you pass the visualizer a pre-fitted model
cm.fit(X_train, y_train)

# To create the ConfusionMatrix, we need some test data. Score runs predict() on the data
# and then creates the confusion_matrix from scikit-learn.
#为了创建ConfusionMatrix，我们需要一些测试数据。对数据执行Score runs predict（）然后从scikit learn创建混淆矩阵。
cm.score(X_test, y_test)

# 图中显示的每一类的个数
cm.show();

