# -*- coding: utf-8 -*-
# @Time : 2024/4/14 16:55
# @Author : nanji
# @Site : 
# @File : testVIF.py
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

boston = load_boston()
X = boston.data
y = boston.target
df = pd.DataFrame(X,columns=boston.feature_names)
df['y'] = boston.target

X = df.iloc[:,:-1]
y = df.iloc[:,-1]
model = sm.OLS(y,X)
model_fit = model.fit()
print(model_fit.summary())

# 定义计算vif的函数
def calculate_vif(df):
    vif = pd.DataFrame()
    vif['index'] = df.columns
    vif['VIF'] = [variance_inflation_factor(df.values,i) for i in range(df.shape[1])]
    return vif

# 使用一个while循环逐步剔除变量

## 先计算每个变量的vif值，再重复计算
vif = calculate_vif(df.iloc[:,:-1])
while (vif['VIF'] > 10).any():
    remove = vif.sort_values(by='VIF',ascending=False)['index'][:1].values[0]
    df.drop(remove,axis=1,inplace=True)
    vif = calculate_vif(df)

feature_selected = vif.iloc[:-1,:]['index'].values
X_new = df[feature_selected]
y_new = y
model = sm.OLS(y_new,X_new)
model_fit = model.fit()
print('2'*100)
print(model_fit.summary())