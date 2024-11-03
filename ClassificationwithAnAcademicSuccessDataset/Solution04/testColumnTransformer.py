# -*- coding: utf-8 -*-
# @Time : 2024/11/3 14:42
# @Author : nanji
# @Site : https://zhuanlan.zhihu.com/p/562538185
# @File : testColumnTransformer.py
# @Software: PyCharm 
# @Comment :
# ColumnTransformer
import pickle

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import (OneHotEncoder, StandardScaler,
                                   KBinsDiscretizer, LabelEncoder, OrdinalEncoder)

data = {
    'age': [10, 15, 12, 45, 36],
    'gender': ['男', '女', '男', '男', '女'],
    'major': ['计算机', '软件工程', '物理', '计算机', '数学']}
data = pd.DataFrame(data)
ct = ColumnTransformer([
    ('cate_feat', OneHotEncoder(sparse=False), ['gender', 'major'])
])
ct.fit_transform(data)

ct = ColumnTransformer([
    ('ordinal', OrdinalEncoder(), ['gender']),  # 类似labelencode,给类别特征加编号
    ('onehot', OneHotEncoder(sparse=False), ['major']),  # onehot
    ('discretizer', KBinsDiscretizer(n_bins=3), ['age']),  # 离散化
    ('scale', StandardScaler(), ['age']),  # 标准化
])
print('0' * 100)
print(ct.fit_transform(data))

ct = ColumnTransformer([
    ('cate_feat', OneHotEncoder(sparse=False), ['gender', 'major']),
    ('pass', 'passthrough', ['age'])
])
res = ct.fit_transform(data)
print('1' * 100)
print(res)

print('2' * 100)
file_path = './.pickle'
# 保存
with open(file_path, 'wb') as f:
    pickle.dump(ct, f)
# 加载
with open(file_path, 'rb') as f:
    ct = pickle.load(f)
