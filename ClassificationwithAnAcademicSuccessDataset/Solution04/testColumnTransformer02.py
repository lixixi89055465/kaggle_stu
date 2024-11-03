# -*- coding: utf-8 -*-
# @Time : 2024/11/3 22:07
# @Author : nanji
# @Site : https://zhuanlan.zhihu.com/p/267522826
# @File : testColumnTransformer02.py
# @Software: PyCharm 
# @Comment :

seed = 123
import pandas as pd
from seaborn import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
import numpy as np

# 加载数据
df = load_dataset('tips').drop(columns=['tip', 'sex']).sample(n=5, random_state=seed)
# 添加缺失的值
df.iloc[[1, 2, 4], [2, 4]] = np.nan
print(df)
# 划分数据
X_Train, X_test, y_train, y_test = train_test_split(
    df.drop(columns=['total_bill', 'size']),
    df['total_bill'],
    test_size=.2,
    random_seed=seed
)
imputer = SimpleImputer(strategy='constant', fill_value='missing')
X_train_imputed = imputer.fit_transform(X_Train)
# 编码训练数据
encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
X_train_encoded = encoder.fit_transform(X_train_imputed)
# 检查训练前后的数据
print('-**********************************Training data ************')

