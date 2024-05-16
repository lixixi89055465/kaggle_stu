# -*- coding: utf-8 -*-
# @Time : 2024/5/16 10:18
# @Author : nanji
# @Site : 
# @File : testLightgbm.py
# @Software: PyCharm 
# @Comment :
import lightgbm as lgb
import numpy as np
import pandas as pd


X = pd.DataFrame({
    "x1": np.random.random(100),
    "x_2": np.random.random(100)
})

y = np.random.random(100)

reg = lgb.LGBMRegressor()
reg.fit(X, y)
