# -*- coding: utf-8 -*-
# @Time : 2024/4/27 22:43
# @Author : nanji
# @Site : https://zhuanlan.zhihu.com/p/595995679
# @File : testoptuna05.py
# @Software: PyCharm 
# @Comment :
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score

# 一，准备数据
data, target = make_classification(n_samples=3000, \
								   n_features=20, \
								   n_informative=12, \
								   n_redundant=4, \
								   n_repeated=0, \
								   n_classes=2, \
								   n_clusters_per_class=4)
x_train_val, x_test, y_train_val, y_test = train_test_split(data, target)
x_train, x_valid, y_train, y_valid = train_test_split(x_train_val, y_train_val)
# 二，训练3个基础模型
tree = DecisionTreeClassifier()
mlp = MLPClassifier()
svc = SVC(probability=True)

mlp.fit(x_train, y_train)
tree.fit(x_train, y_train)
svc.fit(x_train, y_train)


# 三，评估模型效果

def get_val_auc(model):
	probs = model.predict_proba(x_valid)[:, 1]
	val_auc = roc_auc_score(y_valid, probs)
	return val_auc


def get_test_auc(model):
	probs = model.predict_proba(x_test)[:, 1]
	val_auc = roc_auc_score(y_test, probs)
	return val_auc


print("mlp_score:", get_test_auc(mlp))
print("tree_score:", get_test_auc(tree))
print("svc_score:", get_test_auc(svc))
import optuna

preds_val = {name: (eval(name)).predict_proba(x_valid)[:, 1] \
			 for name in ['mlp', 'tree', 'svc']}


def objective(trial):
	weights = {name: trial.suggest_int(name, 1, 100) for name in ['mlp', 'tree', 'svc']}
	probs = sum([weights[name] * preds_val[name] for name in ['mlp', 'tree', 'svc']]) / sum(
		[weights[name] for name in ['mlp', 'tree', 'svc']])
	val_auc = roc_auc_score(y_valid, probs)
	trial.report(val_auc, 0)
	return val_auc


storage_name = 'sqlite:///optuna.db'
study = optuna.create_study(
	direction='maximize', \
	study_name='optuna_ensemble', \
	storage=storage_name, \
	load_if_exists=True
)
study.optimize(objective, n_trials=300, timeout=600)

best_params = study.best_params
best_value = study.best_value
print('\n best_value = ' + str(best_value))
print('best_params: ')
print(best_params)
# best_params:
# {'mlp': 93, 'svc': 81, 'tree': 1}
# 五， 评估模型融合效果

preds_test = {name: (eval(name)).predict_proba(x_test)[:, 1] for name in ['mlp', 'tree', 'svc']}


def atest_score(weights):
	probs = sum([weights[name] * preds_test[name] for name in ['mlp', 'tree', 'svc']]) / sum(
		[weights[name] for name in ['mlp', 'tree', 'svc']]
	)
	test_auc = roc_auc_score(y_test, probs)
	return test_auc


# optuna_ensemble_score: 0.955444558287796


print('optuna_ensembel _score :', atest_score(best_params))
