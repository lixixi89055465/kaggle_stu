# -*- coding: utf-8 -*-
# @Time : 2024/6/3 11:19
# @Author : nanji
# @Site : https://www.kaggle.com/code/gauravduttakiit/pss4e6-flaml-roc-auc-ovo
# @File : solve02.py
# @Software: PyCharm 
# @Comment :
import numpy as np
import pandas as pd
from datetime import datetime

train = pd.read_csv('../input/playground-series-s4e6/train.csv')
# print(train.head())
test = pd.read_csv('../input/playground-series-s4e6/test.csv')
# print(test.head())
print('0' * 100)
# print(train.info())
print('1' * 100)
# print(train.nunique())

r1 = round(train['Target'].value_counts() * 100 / len(train), 2)
print(r1)
print('2' * 100)
print(train.isnull().sum())
import re

train = train.rename(columns=lambda x: re.sub('[^A-Za-z0-9_]+', '', x))
print('3' * 100)
print(train.head())

test = test.rename(columns=lambda x: re.sub('[^A-Za-z0-9_]+', '', x))
print(test.head())


def reduce_mem_usage(df):
	""" iterate through all the columns of a dataframe and modify the data type
	    to reduce memory usage.
	"""
	start_mem = df.memory_usage().sum() / 1024 ** 2
	print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))
	for col in df.columns:
		col_type = df[col].dtype
		if col_type != object:
			c_min = df[col].min()
			c_max = df[col].max()
			if str(col_type)[:3] == 'in':
				if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
					df[col] = df[col].astype(np.int8)
				elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
					df[col] = df[col].astype(np.int16)
				elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
					df[col] = df[col].astype(np.int32)
				elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
					df[col] = df[col].astype(np.int32)
			else:
				if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
					df[col] = df[col].astype(np.float16)
				elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
					df[col] = df[col].astype(np.float32)
				else:
					df[col] = df[col].astype(np.float64)
		else:
			df[col] = df[col].astype('object')
	end_mem = df.memory_usage().sum() / 1024 ** 2
	print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
	print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
	return df


train = reduce_mem_usage(train)
test = reduce_mem_usage(test)

from flaml import AutoML

automl = AutoML()
y = train.pop('Target')
X = train
print(f'{datetime.now()} automl start !')
automl.fit(X, y, task='classification', metric='roc_auc_ovo', time_budget=3600 * 3)
print(f'{datetime.now()} automl end !')

y_pred = automl.predict(test)
print('y_pred[:5]:')
print(y_pred[:5])
df = pd.DataFrame(y_pred, columns=['Target'])
print('df.head():')
print(df.head())

sol = pd.read_csv('../input/playground-series-s4e6/sample_submission.csv')
print('sol.head():')
print(sol.head())
sol.to_csv('./roc_auc_ovo.csv', index=False)
