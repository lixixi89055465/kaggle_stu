# -*- coding: utf-8 -*-
# @Time : 2024/4/27 11:01
# @Author : nanji
# @Site :https://zhuanlan.zhihu.com/p/159506313
# @File : testoptuna02.py
# @Software: PyCharm 
# @Comment :
from hyperopt import hp
from hyperopt import fmin, tpe, space_eval, Trials


def objective(args):
	return -(args['x'] + args['y']) ** 2


space = {
	'x': hp.uniform('x', -10, 10),
	'y': hp.uniform('y', -10, 10)
}
trias = Trials()
best = fmin(objective, space, algo=tpe.suggest, max_evals=100, trials=trias)
print(best)
print(space_eval(space, best))
study_name = 'example-study'  # 不同的 study 不能使用相同的名字。因为当存储在同一个数据库中时，这是区分不同 study 的标识符.
study = optuna.create_study(study_name=study_name, storage='sqlite:///example.db')

study.optimize(objective, n_trials=300)