# -*- coding: utf-8 -*-
# @Time : 2024/4/8 15:07
# @Author : nanji
# @Site : https://mp.weixin.qq.com/s/iYumC_JlMHZpBuAd4ryWFw
# @File : testcatboost01.py
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
X_train, X_validation, y_train, y_validation = train_test_split(X, y, train_size=0.75, random_state=42)
X_test = test_df

from catboost import CatBoostClassifier,Pool,metrics,cv
from sklearn.metrics import accuracy_score
model=CatBoostClassifier(
	custom_loss=[metrics.Accuracy()],
	random_seed=42,
	logging_level='silent'
)
categorical_features_indices=[]
model.fit(
	X_train,y_train,
	cat_features=categorical_features_indices,
	eval_set=(X_validation,y_validation),
	plot=True,
)

cv_params = model.get_params()
cv_params.update({
    'loss_function': metrics.Logloss()
})
cv_data = cv(
    Pool(X, y, cat_features=categorical_features_indices),
    cv_params,
    plot=True
)