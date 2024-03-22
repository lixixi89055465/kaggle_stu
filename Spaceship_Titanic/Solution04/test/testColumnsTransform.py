# -*- coding: utf-8 -*-
# @Time : 2024/3/22 11:13
# @Author : nanji
# @Site : 
# @File : testColumnsTransform.py
# @Software: PyCharm 
# @Comment :  https://zhuanlan.zhihu.com/p/562538185

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder,StandardScaler,KBinsDiscretizer,LabelEncoder,OrdinalEncoder

data = {
    'age': [10, 15, 12, 45, 36],
    'gender': ['男', '女', '男', '男', '女'],
    'major': ['计算机', '软件工程', '物理', '计算机', '数学']}
data = pd.DataFrame(data)
print(data)

ct=ColumnTransformer(transformers=[
    ('cate_feat',OneHotEncoder(sparse=False),['gender','major']),
])
y=ct.fit_transform(data)
print(y)
ct = ColumnTransformer([
    ('ordinal', OrdinalEncoder(), ['gender']),                  # 类似labelencode,给类别特征加编号
    ('onehot', OneHotEncoder(sparse=False), ['major']),         # onehot
    ('discretizer', KBinsDiscretizer(n_bins=3), ['age']),       # 离散化
    ('scale', StandardScaler(), ['age']),                       # 标准化
])
y=ct.fit_transform(data)
print('0'*100)
print(y)