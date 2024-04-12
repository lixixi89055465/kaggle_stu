# -*- coding: utf-8 -*-
# @Time : 2024/4/11 11:26
# @Author : nanji
# @Site : https://mp.weixin.qq.com/s?__biz=MzkxODI5NjE1OQ==&mid=2247497152&idx=1&sn=e60474f073b257f53fae1aed4b647752&chksm=c1b136f0f6c6bfe61c1e88d1662ff47c9fd325b6d07c5c47021b927ff4e2124e57a0190831d4&scene=21#wechat_redirect
# @File : testCatBoost04.py
# @Software: PyCharm 
# @Comment :

import pandas as pd
import os
import numpy as np
np.set_printoptions(precision=4)
import catboost
from catboost import *
from catboost import datasets

(train_df, test_df) = catboost.datasets.amazon()
print(train_df.head())
y = train_df.ACTION
X = train_df.drop('ACTION', axis=1)
print('0'*100)
print(X.shape)

print('Label :{}'.format(set(y)))
print('1'*100)
print('Zero count= {} ,One count= {}'.format(len(y)-sum(y),sum(y)))

dataset_dir = './amazon'
if not os.path.exists(dataset_dir):
	os.makedirs(dataset_dir)

train_df.to_csv(
	os.path.join(dataset_dir, 'train.csv'),
	index=False, sep=',', header=True
)
test_df.to_csv(
	os.path.join(dataset_dir, 'test.csv'),
	index=False, sep=',', header=True
)

from catboost.utils import create_cd

feature_names = dict()
for column, name in enumerate(train_df):
	if column == 0:
		continue
	feature_names[column - 1] = name

create_cd(
	label=0,
	cat_features=list(range(1, train_df.columns.shape[0])),
	feature_names=feature_names,
	output_path=os.path.join(dataset_dir, 'train.cd')
)


