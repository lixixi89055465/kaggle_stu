# -*- coding: utf-8 -*-
# @Time : 2024/4/28 22:08
# @Author : nanji
# @Site :https://blog.csdn.net/WHYbeHERE/article/details/135483560
# @File : testoptuna06.py
# @Software: PyCharm 
# @Comment : 

import optuna
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from lightgbm import LGBMClassifier

# 载入数据
data = datasets.load_iris()
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2)


# 定义目标函数
def objective(trial):
	params = {
		'objective': 'multiclass',
		'metric': 'multi_logloss',  # Use 'multi_logloss' for evaluation
		'boosting_type': 'gbdt',
		'num_class': 3,  # Replace with the actual number of classes
		'num_leaves': trial.suggest_int('num_leaves', 2, 256),
		'learning_rate': trial.suggest_loguniform('learning_rate', 0.001, 0.1),
		'feature_fraction': trial.suggest_uniform('feature_fraction', 0.1, 1.0),
		'bagging_fraction': trial.suggest_uniform('bagging_fraction', 0.1, 1.0),
		'bagging_freq': trial.suggest_int('bagging_freq', 1, 10),
		'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
	}
	model = LGBMClassifier(**params)
	model.fit(X_train, y_train)
	# y_pred=model.predict_proba(X_val)
	# loss=log_loss(y_val,y_pred)
	# return loss


study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50,show_progress_bar=True)

# 获取最佳超参数
best_params = study.best_params
print(f"最佳超参数：{best_params}")
