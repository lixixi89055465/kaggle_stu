# -*- coding: utf-8 -*-
# @Time : 2024/3/30 21:17
# @Author : nanji
# @Site :  https://blog.csdn.net/ab1112221212/article/details/100133066
# @File : teststatsmodels02.py
# @Software: PyCharm 
# @Comment :
import numpy as np
def get_var_no_colinear(cutoff, df):
    corr_high = df.corr().applymap(lambda x: np.nan if x>cutoff else x).isnull()
    col_all = corr_high.columns.tolist()
    del_col = []
    i = 0
    while i < len(col_all)-1:
        ex_index = corr_high.iloc[:,i][i+1:].index[np.where(corr_high.iloc[:,i][i+1:])].tolist()
        for var in ex_index:
            col_all.remove(var)
        corr_high = corr_high.loc[col_all, col_all]
        i += 1
    return col_all
