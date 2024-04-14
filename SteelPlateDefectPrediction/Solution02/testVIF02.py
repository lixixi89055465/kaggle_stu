# -*- coding: utf-8 -*-
# @Time : 2024/4/14 17:25
# @Author : nanji
# @Site : https://zhuanlan.zhihu.com/p/435430759
# @File : testVIF02.py
# @Software: PyCharm 
# @Comment :

import numpy as np
import pandas as pd
from sklearn.datasets import load_boston
from sklearn.linear_model import LogisticRegression
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore')
print('0'*100)
from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()

X = cancer.data
y = cancer.target
df = pd.DataFrame(X,columns=cancer.feature_names)
print('1'*100)
print(df.shape)
df['y'] = cancer.target

X = df.iloc[:,:-1]
y = df.iloc[:,-1]
model = LogisticRegression()
model.fit(X,y)
print(model.score(X, y))

def calculate_vif(df):
    vif = pd.DataFrame()
    vif['index'] = df.columns
    vif['VIF'] = [variance_inflation_factor(df.values,i) for i in range(df.shape[1])]
    return vif
# 以VIF30为阈值
vif = calculate_vif(df.iloc[:,:-1])

while (vif['VIF'] > 30).any():
    remove = vif.sort_values(by='VIF',ascending=False)['index'][:1].values[0]
    df.drop(remove,axis=1,inplace=True)
    vif = calculate_vif(df)

feature_selected = vif.iloc[:-1,:]['index'].values
X_new = df[feature_selected]
y_new = y

model = LogisticRegression()
model.fit(X_new,y_new)
print(model.score(X_new, y_new))
print('3'*100)
print(X_new.shape)