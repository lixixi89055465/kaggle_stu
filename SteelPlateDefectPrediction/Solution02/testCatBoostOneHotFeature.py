# -*- coding: utf-8 -*-
# @Time : 2024/4/9 10:53
# @Author : nanji
# @Site : https://mp.weixin.qq.com/s?__biz=MzkxODI5NjE1OQ==&mid=2247497152&idx=1&sn=e60474f073b257f53fae1aed4b647752&chksm=c1b136f0f6c6bfe61c1e88d1662ff47c9fd325b6d07c5c47021b927ff4e2124e57a0190831d4&scene=21#wechat_redirect
# @File : testCatBoostOneHotFeature.py
# @Software: PyCharm 
# @Comment :
from catboost.datasets import titanic
from catboost import Pool, CatBoostClassifier
import warnings
warnings.filterwarnings("ignore")
titanic_df = titanic()
print(titanic_df[0].columns)
X = titanic_df[0].drop('Survived', axis=1)
y = titanic_df[0].Survived
print(X.head())
cat_features_index = []
pool = Pool(X, y, cat_features=cat_features_index, feature_names=list(X.columns))
model = CatBoostClassifier(
	max_depth=2, verbose=False, max_ctr_complexity=1, random_seed=42, iterations=2).fit(pool)
# model.fit(X,y,plot=True)
model.plot_tree(
	tree_idx=0,
	pool=pool,
)


