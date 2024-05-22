# -*- coding: utf-8 -*-
# @Time : 2024/5/22 17:53
# @Author : nanji
# @Site : https://zhuanlan.zhihu.com/p/562538185
# @File : testColumnTransformer01.py
# @Software: PyCharm 
# @Comment :

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, \
	StandardScaler, \
	KBinsDiscretizer, \
	LabelEncoder, \
	OrdinalEncoder

data = {
	'age': [10, 15, 12, 45, 36],
	'gender': ['男', '女', '男', '男', '女'],
	'major': ['计算机', '软件工程', '物理', '计算机', '数学']}
data = pd.DataFrame(data)
# ct = ColumnTransformer([
# 	('cate_feat', OneHotEncoder(sparse=False), ['gender', 'major'])
# ])
# r1 = ct.fit_transform(data)
# print(r1)
# ct = ColumnTransformer(
# 	transformers=[
# 		('ordinal', OrdinalEncoder(), ['gender']),  # 类似labelencode,给类别特征加编号
# 		('onehot', OneHotEncoder(sparse=False), ['major']),  # onehot
# 		('discretizer', KBinsDiscretizer(n_bins=3), ['age']),  # 离散化
# 		('scale', StandardScaler(), ['age']),
# 	],
# )
# r2 = ct.fit_transform(data)
# print('0'*100)
# print(r2)

ct = ColumnTransformer([
    ('cate_feat', OneHotEncoder(sparse=False), ['gender', 'major']),
    ('pass', 'passthrough', ['age'])
])
r3=ct.fit_transform(data)
print(r3)