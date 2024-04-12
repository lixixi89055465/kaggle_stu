# -*- coding: utf-8 -*-
# @Time : 2024/4/12 13:58
# @Author : nanji
# @Site : https://mp.weixin.qq.com/s/v54P5kBiMk3GL4esuut46w
# @File : testUsedcarscatalog.py
# @Software: PyCharm 
# @Comment :
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('cars.csv')
print('0' * 100)
print(df.head())
categorical_features_names = ['manufacturer_name', 'model_name', 'transmission',
							  'color', 'engine_fuel', 'engine_type', 'body_type',
							  'state', 'drivetrain', 'location_region']
print('1' * 100)
print(df[categorical_features_names].nunique())
print('2' * 100)
import seaborn as sns
import numpy as np

ax = sns.distplot(df.price_usd.values)
print('3' * 100)
print(np.median(df.price_usd.values))
plt.show()
from catboost import CatBoost, CatBoostRegressor, Pool

df_ = df.sample(frac=1, random_state=0)
df_train = df_.iloc[:2 * len(df) // 3]
df_test = df_.iloc[2 * len(df) // 3:]

train_pool = Pool(df_train.drop(['price_usd'], axis=1), \
				  label=df_train.price_usd, \
				  cat_features=categorical_features_names)
test_pool = Pool(df_test.drop(['price_usd'], axis=1), \
				 label=df_test.price_usd, \
				 cat_features=categorical_features_names)
model = CatBoostRegressor(
	custom_metric=['R2', 'RMSE'], \
	learning_rate=0.1, \
	n_estimators=5000)
model.fit(train_pool, \
		  eval_set=test_pool, \
		  verbose=2000, plot=True)

from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_squared_error

rmse_values = []
kf = KFold(n_splits=3, shuffle=True)
default_parameters = {'n_estimators': 4500, \
					  'learning_rate': 0.1}
default_model_metrics = {}


def score_catboost_model(catboost_parameters, update_parameters=False):
	r2_values = []
	rmse_values = []
	catboost_parameters.update(default_parameters)
	for train_index, test_index in kf.split(df):
		train_pool = Pool(
			df.iloc[train_index].drop(['price_usd'], axis=1), \
			label=df.iloc[train_index].price_usd, \
			cat_features=categorical_features_names)
		test_pool = Pool(df.iloc[test_index].drop(['pricec_usd'], axis=1), \
						 label=df.iloc[test_index].price_usd, \
						 cat_features=categorical_features_names)
		model = CatBoost(catboost_parameters)
		model.fit(train_pool, verbose=False)
		r2_values.append(r2_score(df.iloc[test_index].price_usd.values, model.predict(test_pool)))
		rmse_values.append(mean_squared_error(df.iloc[test_index].price_usd.values, \
											  model.predict(test_pool), \
											  squared=False))
