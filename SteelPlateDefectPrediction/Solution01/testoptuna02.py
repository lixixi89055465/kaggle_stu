# -*- coding: utf-8 -*-
# @Time : 2024/3/28 11:03
# @Author : nanji
# @Site : https://optuna.readthedocs.io/zh-cn/latest/index.html
# @File : testoptuna02.py
# @Software: PyCharm 
# @Comment :

import optuna
import warnings
import sklearn
from sklearn import svm,datasets,metrics,ensemble
warnings.filterwarnings('ignore')

# Define an objective function to be minimized.
def objective(trial):
	# Invoke suggest methods of a Trial object to generate hyperparameters.
	regressor_name = trial.suggest_categorical('classifier', ['SVR', 'RandomForest'])
	if regressor_name == 'SVR':
		svr_c = trial.suggest_float('svr_c', 1e-10, 1e10, log=True)
		regressor_obj = svm.SVR(C=svr_c)
	else:
		rf_max_depth = trial.suggest_int('rf_max_depth', 2, 32)
		regressor_obj = ensemble.RandomForestRegressor(max_depth=rf_max_depth)

	X, y = datasets.load_boston(return_X_y=True)
	X_train, X_val, y_train, y_val = sklearn.model_selection.train_test_split(X, y, random_state=0)

	regressor_obj.fit(X_train, y_train)
	y_pred = regressor_obj.predict(X_val)

	error = metrics.mean_squared_error(y_val, y_pred)

	return error  # An objective value linked with the Trial object.


study = optuna.create_study()  # Create a new study.
study.optimize(objective, n_trials=100)  # Invoke optimization of the objective function.
