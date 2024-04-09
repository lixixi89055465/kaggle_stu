# -*- coding: utf-8 -*-
# @Time : 2024/4/9 11:07
# @Author : nanji
# @Site : https://mp.weixin.qq.com/s?__biz=MzkxODI5NjE1OQ==&mid=2247497152&idx=1&sn=e60474f073b257f53fae1aed4b647752&chksm=c1b136f0f6c6bfe61c1e88d1662ff47c9fd325b6d07c5c47021b927ff4e2124e57a0190831d4&scene=21#wechat_redirect
# @File : testCatBoost03.py
# @Software: PyCharm 
# @Comment :

from catboost.datasets import titanic
import numpy as np

train_df, test_df = titanic()

# 用一些超出分布范围的数字来填充缺失值
train_df.fillna(-999, inplace=True)
test_df.fillna(-999, inplace=True)

# 拆分特征变量及标签变量
X = train_df.drop('Survived', axis=1)
y = train_df.Survived
# 划分训练集和测试集

from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=.75, random_state=42)
print(X_train.columns)
print('0'*100)
print(X_train.info())
print(X.head(10))
X_test = test_df

from catboost import CatBoostClassifier, Pool, metrics, cv
from sklearn.metrics import accuracy_score

model = CatBoostClassifier(
	custom_loss=[metrics.Accuracy()], \
	random_state=42, \
	logging_level='silent' \
	)
categorical_features_indices=['Pclass','Embarked']
# 模型训练
model.fit(
	X_train,y_train,
	# cat_features=categorical_features_indices,
	eval_set=(X_val, y_val),
	plot=True

)
