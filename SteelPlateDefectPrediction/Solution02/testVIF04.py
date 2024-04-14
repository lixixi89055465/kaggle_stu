# -*- coding: utf-8 -*-
# @Time : 2024/4/14 21:10
# @Author : nanji
# @Site : https://zhuanlan.zhihu.com/p/666073993
# @File : testVIF04.py
# @Software: PyCharm 
# @Comment :

import pandas as pd
import numpy as np
from statsmodels.stats.outliers_influence import variance_inflation_factor

bmi = pd.read_csv('bmi.csv')
print(bmi.head())
bmi['Gender'] = bmi['Gender'].map({'Male': 0, 'Female': 1})

correlation_matrix = bmi.corr()
eigenvalues = np.linalg.eigvals(correlation_matrix)
condition_index = np.sqrt(max(eigenvalues) / eigenvalues)
print(f'Condition Index:{condition_index}')
print('0'*100)
bmi_2 = bmi.drop(['Index', 'Weight'], axis=1)
vif = pd.DataFrame()
vif["VIF Factor"] = [variance_inflation_factor(bmi_2.values, i) for i in range(bmi_2.shape[1])]
vif['features']=bmi_2.columns
print('1'*100)
print(vif)


