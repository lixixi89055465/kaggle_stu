# -*- coding: utf-8 -*-
# @Time : 2024/4/27 10:56
# @Author : nanji
# @Site :https://zhuanlan.zhihu.com/p/159506313
# @File : testoptuna01.py
# @Software: PyCharm 
# @Comment :
import optuna


def objective(trial):
	x = trial.suggest_uniform('x', -9, 9)
	y = trial.suggest_uniform('y', -9, 9)
	return (x + y) ** 2


study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)
print('0'*100)
print(study.best_params)
print('1'*100)
print(study.best_value)
