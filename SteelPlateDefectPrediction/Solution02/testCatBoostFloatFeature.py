# -*- coding: utf-8 -*-
# @Time : 2024/4/9 10:47
# @Author : nanji
# @Site : https://mp.weixin.qq.com/s?__biz=MzkxODI5NjE1OQ==&mid=2247497152&idx=1&sn=e60474f073b257f53fae1aed4b647752&chksm=c1b136f0f6c6bfe61c1e88d1662ff47c9fd325b6d07c5c47021b927ff4e2124e57a0190831d4&scene=21#wechat_redirect
# @File : testCatBoostFloatFeature.py
# @Software: PyCharm 
# @Comment :

from catboost.datasets import titanic
import catboost
from catboost import CatBoostRegressor
import numpy as np
from sklearn.datasets import load_boston
import warnings

warnings.filterwarnings('ignore')

boston = load_boston()
y = boston['target']
x = boston['data']
pool = catboost.Pool(data=x, label=y)
model = CatBoostRegressor(depth=2, verbose=False, iterations=1).fit(x, y)
model.plot_tree(tree_idx=0, \
				# pool=pool
				)
