# -*- coding: utf-8 -*-
# @Time : 2024/3/24 14:55
# @Author : nanji
# @Site : https://zhuanlan.zhihu.com/p/260701846
# @File : teststatsmodels.py
# @Software: PyCharm 
# @Comment : 

import statsmodels.formula.api as smf
import pandas as pd
import numpy as np
data=pd.DataFrame({'X':np.arange(10,20,0.25)})
data['Y']=2*data['X']+1+np.random.randn(40)
mod=smf.ols('Y ~ X',data).fit()
print(mod.summary())
print('0'*100)
import matplotlib.pyplot as plt
data.plot(x="X", y="Y",kind="scatter",figsize=(8,5))
plt.plot(data["X"], mod.params[0] + mod.params[1]*data["X"],"r")
plt.text(10, 38, "y="+str(round(mod.params[1],4)) + "*x" + str(round(mod.params[0],4)))
plt.title("linear regression")
plt.show()
