# -*- coding: utf-8 -*-
# @Time : 2024/4/14 20:12
# @Author : nanji
# @Site : https://zhuanlan.zhihu.com/p/666073993
# @File : testVIF03.py
# @Software: PyCharm 
# @Comment : 
from statsmodels.stats.outliers_influence import variance_inflation_factor

import pandas as pd

bmi = pd.read_csv('bmi.csv')
print(bmi.head())

bmi['Gender'] = bmi['Gender'].map({'Male': 0, 'Female': 1})
vif = pd.DataFrame()
vif['features'] = bmi.columns
vif['VIF Factor'] = [variance_inflation_factor(bmi.values, i) for i in range(bmi.shape[1])]
print('0'*100)
print(vif)
correlation_matric=bmi.corr()
print('1'*100)
print(correlation_matric)
import matplotlib.pyplot as plt
import seaborn as sns
plt.figure(figsize=(6,5))
sns.heatmap(correlation_matric,annot=True)
plt.title('correlation matrix')
plt.show()