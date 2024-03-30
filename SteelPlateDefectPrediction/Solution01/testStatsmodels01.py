# -*- coding: utf-8 -*-
# @Time : 2024/3/30 17:37
# @Author : nanji
# @Site : https://zhuanlan.zhihu.com/p/446585802
# @File : testStatsmodels01.py
# @Software: PyCharm 
# @Comment :

from statsmodels.compat import lzip

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import statsmodels.stats.api as sms
import matplotlib.pyplot as plt

#load data
url='Guerry.csv'
dat=pd.read_csv(url)
# Fit regression model (using the natural log of one of the regressors)
results=smf.ols("Lottery ~ Literacy + np.log(Pop1831)", data=dat).fit()
# Inspect the results
print('0'*100)

print(results.summary())

print('1'*100)
name = ["Jarque-Bera", "Chi^2 two-tail prob.", "Skew", "Kurtosis"]
test=sms.jarque_bera(results.resid)
results01=lzip(name,test)
print(results01)
