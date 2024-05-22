# -*- coding: utf-8 -*-
# @Time : 2024/5/6 22:08
# @Author : nanji
# @Site :https://blog.csdn.net/m0_55894587/article/details/130474769
# @File : testPowerTransformer.py
# @Software: PyCharm 
# @Comment : 
# import warnings
# warnings.filterwarnings('ignore')
import numpy, warnings
numpy.warnings = warnings
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB

df = pd.read_csv('../input/heart.csv')
print(df.isna().sum())
df['gender'] = LabelEncoder().fit_transform(df['gender'])
print(df['class'].value_counts())

print(df['class'].value_counts())
# 对数据进行简单的过采样
df_1 = df[df['class'] == 1]
# df = pd.concat([df, df_1])
# df = pd.concat([df, df_1])
print('0' * 100)
print(df['class'].value_counts())

X = df.iloc[:, :-1]  # 特征列
Y = df.iloc[:, -1:].values.ravel()  # 目标列
print(X.shape)
print(Y.shape)

# 直接划分训练集与测试集进行分类
X_train, X_test, y1_train, y1_test = train_test_split(X, Y, test_size=0.2, random_state=0)
gnb1 = GaussianNB()
gnb1.fit(X_train, y1_train)
pre = gnb1.predict(X_test)
pre = pre.astype(np.int64)
acc_score = round(accuracy_score(pre, y1_test), 4)
print('未进行正态化处理的准确率：{}'.format(acc_score))

# # 对数据进行正态化处理
from sklearn.preprocessing import PowerTransformer
powerTransformer=PowerTransformer()
powerTransformer.fit(X)
px=powerTransformer.transform(X)
px_train,px_test,y2_train,y2_test=train_test_split(px,Y,\
												   test_size=0.2, \
												   random_state=0)
# 对正态化处理的数据进行分类

gnb2=GaussianNB()
gnb2.fit(px_train,y2_train)
pre=gnb2.predict(px_test)
pre=pre.astype(np.int64)
acc_score=round(accuracy_score(pre,y2_test),4)

print('进行正态化处理准确率：{}'.format(acc_score))
