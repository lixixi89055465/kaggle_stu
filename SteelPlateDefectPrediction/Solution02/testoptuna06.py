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
# 载入数据
data = datasets.load_iris()
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2)
# 定义目标函数
def objective(trial):
	# 定义超参数搜索范围
	C = trial.suggest_loguniform('C', 1e-5, 1e5)
	gamma = trial.suggest_loguniform('gamma', 1e-5, 1e5)
	# 构建SVM模型
	model = SVC(C=C, gamma=gamma)

	# 训练和评估模型
	model.fit(X_train, y_train)
	accuracy = model.score(X_test, y_test)
	return accuracy
study=optuna.create_study(direction='maximize')
study.optimize(objective,n_trials=100)

# 获取最佳超参数
best_params=study.best_params
print("最佳超参数：", best_params)








