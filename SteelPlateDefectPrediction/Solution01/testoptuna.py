# -*- coding: utf-8 -*-
# @Time : 2024/3/27 23:25
# @Author : nanji
# @Site : https://zhuanlan.zhihu.com/p/159506313
# @File : testoptuna.py
# @Software: PyCharm 
# @Comment : 
import optuna

def objective(trial):
    x = trial.suggest_uniform('x', -10, 10)
    y = trial.suggest_uniform('y', -10, 10)
    return (x + y) ** 2

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)

print(study.best_params)
print(study.best_value)