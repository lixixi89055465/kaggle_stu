# -*- coding: utf-8 -*-
# @Time : 2024/5/1 10:59
# @Author : nanji
# @Site : 
# @File : test08.py
# @Software: PyCharm 
# @Comment : 
import optuna


def objective(trial):
	x = trial.suggest_float("x", -1, 1)
	y = trial.suggest_int("y", -1, 1)
	return x ** 2 + y


sampler = optuna.samplers.CmaEsSampler()
study = optuna.create_study(sampler=sampler)
study.optimize(objective, n_trials=20)