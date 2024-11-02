# -*- coding: utf-8 -*-
# @Time : 2024/11/2 21:28
# @Author : nanji
# @Site : https://blog.csdn.net/wang2leee/article/details/134026432
# @File : testchi2_contingency.py
# @Software: PyCharm 
# @Comment :
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency

# table = np.array([[176, 230], [21035, 21018]])
# res = chi2_contingency(table)
# print(res.statistic)
# print('0'*100)
# print(res.pvalue)
obs = np.array([[10, 10, 20], [20, 20, 20]])
res = chi2_contingency(obs)
print(res)


def chi_test(x, y):
    from scipy.stats import chi2_contingency
    tab = pd.crosstab(x, y).fillna(0)
    chi_value, p_value, def_free, _ = chi2_contingency(tab)
    return {'DF': def_free, 'Value': chi_value, 'Prob': p_value,
            'LIP': -np.log(p_value)}

chi_df_1 = pd.DataFrame(columns=
                        ['name1','name2','chi_value','chi_p','DF','LLP'])
#首先创造一个空的表，用于下面的append使用
for i in range(2,17): #这个是类别变量位置
    for j in range(i+1,17):  #主要为了不重复计算
        chi_df=pd.DataFrame()
        chi_df['name1'] = [list(main_people.columns)[i]]  #记住要lis，不能是string
        chi_df['name2'] = [list(main_people.columns)[j]]


