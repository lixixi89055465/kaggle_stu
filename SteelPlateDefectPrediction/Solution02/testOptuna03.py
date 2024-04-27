# -*- coding: utf-8 -*-
# @Time : 2024/4/27 18:03
# @Author : nanji
# @Site :https://zhuanlan.zhihu.com/p/159506313
# @File : testOptuna03.py
# @Software: PyCharm 
# @Comment : 
import optuna

def objective(trial):
    x = trial.suggest_uniform('x', -10, 10)
    y = trial.suggest_uniform('y', -10, 10)
    return (x + y) ** 2

# study = optuna.create_study(direction='maximize')
# study.optimize(objective, n_trials=100)

# print(study.best_params)
# print(study.best_value)
print('0'*100)
study_name = 'example-study'  # 不同的 study 不能使用相同的名字。因为当存储在同一个数据库中时，这是区分不同 study 的标识符.
study = optuna.create_study(study_name=study_name, storage='sqlite:///example.db')

study.optimize(objective, n_trials=300)
