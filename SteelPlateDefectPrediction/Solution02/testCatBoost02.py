# -*- coding: utf-8 -*-
# @Time : 2024/4/16 14:17
# @Author : nanji
# @Site : https://zhuanlan.zhihu.com/p/475432922
# @File : testCatBoost02.py
# @Software: PyCharm 
# @Comment :

import catboost as cb
import numpy as np

print(cb.__version__)
from IPython.display import display
import datetime, json
import pandas as pd
import catboost as cb
from catboost.datasets import titanic
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
import plotly.graph_objs as go
import plotly.express as px


def printlog(info):
	nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
	print('\n' + '==========' * 8 + '%s ' % nowtime)
	print(info + '...\n\n')


printlog("step1: preparing data...")
dfdata, dftest = titanic()
display(dfdata.head())

label_col = 'Survived'
dfnull = pd.DataFrame(dfdata.isnull().sum(axis=0), columns=['null_cnt']).query('null_cnt>0')
print('null_features:')
print(dfnull)
dfdata.fillna(-9999, inplace=True)
dftest.fillna(-9999, inplace=True)
cate_cols = [x for x in dfdata.columns if dfdata[x].dtype not in [np.float32, np.float64]
			 and x != label_col]
for col in cate_cols:
	dfdata[col] = pd.Categorical(dfdata[col])
	dftest[col] = pd.Categorical(dftest[col])

# 分割数据集
dftrain, dfvalid = train_test_split(dfdata, train_size=0.85, random_state=42)
Xtrain, Ytrain = dftrain.drop(label_col, axis=1), dftrain[label_col]
Xvalid, Yvalid = dfvalid.drop(label_col, axis=1), dfvalid[label_col]
cate_cols_indexs = np.where(Xtrain.columns.isin(cate_cols))[0]
# 整理成Pool
