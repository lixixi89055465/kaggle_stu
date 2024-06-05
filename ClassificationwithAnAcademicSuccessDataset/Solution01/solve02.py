# -*- coding: utf-8 -*-
# @Time : 2024/6/3 11:19
# @Author : nanji
# @Site : https://www.kaggle.com/code/gauravduttakiit/pss4e6-flaml-roc-auc-ovo
# @File : solve02.py
# @Software: PyCharm 
# @Comment :
from sklearn.pipeline import Pipeline, make_pipeline
from tqdm.notebook import tqdm
import numpy as np
import pandas as pd
from datetime import datetime
import optuna
from optuna import Trial, trial, create_study
from optuna.pruners import HyperbandPruner
from optuna.samplers import TPESampler, CmaEsSampler

optuna.logging.set_verbosity = optuna.logging.ERROR
from functools import partial

from sklearn.metrics import accuracy_score, roc_auc_score, make_scorer
from sklearn.model_selection import (RepeatedStratifiedKFold as RSKF, \
									 StratifiedKFold as SKF, \
									 StratifiedGroupKFold as SGKF, \
									 KFold, \
									 RepeatedKFold as RKF, \
									 cross_val_score, \
									 cross_val_predict)
# ML Model training : ~
from sklearn.metrics import accuracy_score, roc_auc_score, make_scorer
from xgboost import DMatrix, XGBClassifier as XGBC
from lightgbm import log_evaluation, early_stopping, LGBMClassifier as LGBMC
from catboost import CatBoostClassifier as CBC, Pool
from sklearn.ensemble import HistGradientBoostingClassifier as HGBC, \
	RandomForestClassifier as RFC
from sklearn.metrics import brier_score_loss

train = pd.read_csv('../input/playground-series-s4e6/train.csv')
# print(train.head())
test = pd.read_csv('../input/playground-series-s4e6/test.csv')
# print(test.head())
print('0' * 100)
# print(train.info())
print('1' * 100)
# print(train.nunique())

r1 = round(train['Target'].value_counts() * 100 / len(train), 2)
print(r1)
print('2' * 100)
print(train.isnull().sum())
import re

train = train.rename(columns=lambda x: re.sub('[^A-Za-z0-9_]+', '', x))
print('3' * 100)
print(train.head())

test = test.rename(columns=lambda x: re.sub('[^A-Za-z0-9_]+', '', x))
print(test.head())


def reduce_mem_usage(df):
	""" iterate through all the columns of a dataframe and modify the data type
	    to reduce memory usage.
	"""
	start_mem = df.memory_usage().sum() / 1024 ** 2
	print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))
	for col in df.columns:
		col_type = df[col].dtype
		if col_type != object:
			c_min = df[col].min()
			c_max = df[col].max()
			if str(col_type)[:3] == 'int':
				if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
					df[col] = df[col].astype(np.int8)
				elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
					df[col] = df[col].astype(np.int16)
				elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
					df[col] = df[col].astype(np.int32)
				elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
					df[col] = df[col].astype(np.int64)
			else:
				if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
					df[col] = df[col].astype(np.float16)
				elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
					df[col] = df[col].astype(np.float32)
				else:
					df[col] = df[col].astype(np.float64)
		else:
			df[col] = df[col].astype('object')
	end_mem = df.memory_usage().sum() / 1024 ** 2
	print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
	print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
	return df


train = reduce_mem_usage(train)
test = reduce_mem_usage(test)


class CFG:
	"""
	Configuration class for parameters and CV strategy for tuning and training
	Some parameters may be unused here as this is a general configuration class
	""";
	# Data preparation
	version_nb = 7
	test_req = 'Y'
	test_sample_frac = 0.025
	gpu_switch = 'OFF'
	state = 42
	targets = ['Target']
	episode = 3
	season = 4
	path = f"../input/playground-series-s{season}e{episode}";
	orig_path = f"../input/playgrounds4e03ancillary/PlaygroundS4E3Original.csv";
	dtl_preproc_req = 'Y'
	ftre_plots_req = 'Y'
	ftre_imp_req = 'Y'

	# Data transforms and scaling
	conjoin_orig_data = 'Y'
	drop_nulls = 'N'
	sec_ftre_req = 'Y'
	scale_req = 'N'
	# Model Training:-
	pstprcs_oof = 'N'
	pstprcs_train = 'N'
	pstprcs_test = 'N'
	ML = 'Y'

	pseudo_lbl_req = 'N'
	pseudolbl_up = 0.975
	pseudolbl_low = 0.00

	# n_splits = 3 if test_req == 'Y' else 10
	n_splits = 3 if test_req == 'Y' else 10
	n_repeats = 1
	nbrnd_erly_stp = 75
	mdlcv_mthd = 'RSKF'

	# Ensemble : -
	ensemble_req = 'Y'
	hill_climb_req = 'N'
	optuna_req = 'Y'
	LAD_req = 'N'
	enscv_mthd = 'RSKF'
	metric_obj = 'maximize'
	ntrials = 10 if test_req == 'Y' else 100
	# Global variables for plotting:-
	grid_specs = {'visible': True, 'which': 'both', 'linestyle': '--',
				  'color': 'lightgrey', 'linewidth': 0.75};
	title_specs = {'fontsize': 9, 'fontweight': 'bold', 'color': '#992600'};


class OptunaEnsembler:
	'''
	  This is the Optuna ensemble class-
    Source- https://www.kaggle.com/code/arunklenin/ps3e26-cirrhosis-survial-prediction-multiclass
	'''

	def __init__(self):
		self.study = None
		self.weights = None
		self.random_state = CFG.state
		self.n_trials = CFG.ntrials
		self.direction = CFG.metric_obj

	def ScoreMetric(self, ytrue, ypred):
		'''
		This is the metric function for the competition
		:param ytrue:
		:param ypred:
		:return:
		'''
		return accuracy_score(ytrue, ypred)

	def _objective(self, trial, y_true, y_preds):
		''''
        This method defines the objective function for the ensemble
		'''
		if isinstance(y_preds, pd.DataFrame) or isinstance(y_preds, np.ndarray):
			weights = [trial.suggest_float(f'weight{n}', 0, 1)
					   for n in range(y_preds.shape[-1])]
			axis = 1
		elif isinstance(y_preds, list):
			weights = [trial.suggest_float(f'weight{n}', 0, 1) \
					   for n in range(len(y_preds))]
			axis = 0
		weighted_pred = np.average(np.array(y_preds), axis=axis, weights=weights)
		score = self.ScoreMetric(y_true, weighted_pred)
		return score

	def fit(self, y_true, y_preds):
		'''
		This method fits the Optuna objective on the fold level data
		:param y_true:
		:param y_preds:
		:return:
		'''
		optuna.logging.set_verbosity = optuna.logging.ERROR
		self.study = optuna.create_study(
			sampler=TPESampler(seed=self.random_state),
			pruner=HyperbandPruner(),
			study_name='Ensemble',
			direction=self.direction
		)
		obj = partial(self._objective, y_true=y_true, y_preds=y_preds)
		self.study.optimize(obj, n_trials=self.n_trials)
		if isinstance(y_preds, list):
			self.weights = [self.study.best_params[f'weight{n}'] \
							for n in range(len(y_preds))]
		else:
			self.weights = [self.study.best_params[f'weight{n}'] \
							for n in range(y_preds.shape[-1])]

	def predict(self, y_preds):
		'''
		This method predicts using the fitted Optuna objective;
		:param y_preds:
		:return:
		'''
		assert self.weights is not None, 'OptunaWeights error, must be fitted before predict';
		if isinstance(y_preds, list):
			weighted_pred = np.average(np.array(y_preds), axis=0, weights=self.weights)
		else:
			weighted_pred = np.average(np.array(y_preds).reshape(-1,1), axis=1, weights=self.weights)
		return weighted_pred

	def fit_predict(self, y_true, y_pres):
		self.fit(y_true, y_pres)
		return self.predict(y_pres)

	def weights(self):
		return self.weights
# Machine Learning Algorithm (MLA) Selection and Initialization
from sklearn import ensemble, gaussian_process, \
	linear_model, naive_bayes, \
	neighbors, svm, tree, discriminant_analysis
from xgboost import XGBClassifier

class MdlDeveloper(CFG):
	'''
	This class implements the training pipeline elements-
    1. Initializes the Model predictions
    2. Trains and infers models
    3. Returns the OOF and model test set predictions
	'''

	def __init__(self, Xtrain, ytrain, ygrp, Xtest, \
				 sel_cols, cat_cols, enc_cols, **kwargs):
		'''
		In this method,we initialize the below -
		1.Train-test data,selected columns
		2.Metric,custom scorer,model and cv object
		4.Output tables for score and predictions
		:param Xtrain:
		:param ytrain:
		:param ygrp:
		:param Xtest:
		:param sel_cols:
		:param cat_cols:
		:param enc_cols:
		:param kwargs:
		'''
		self.Xtrain = Xtrain
		self.ytrain = ytrain
		self.y_grp = ygrp
		self.Xtest = Xtest
		self.sel_cols = sel_cols
		self.cat_cols = cat_cols
		self.enc_cols = enc_cols
		self._DefineModel()
		self.cv = self.all_cv[self.mdlcv_mthd]
		self.methods = list(self.Mdl_Master.keys())
		self.OOF_Preds = pd.DataFrame()
		self.Mdl_Preds = pd.DataFrame()
		self.Scores = pd.DataFrame( \
			columns=self.methods + ['Ensemble'], \
			index=range(self.n_splits * self.n_repeats))
		self.TrainScores = pd.DataFrame( \
			columns=self.methods, \
			index=range(self.n_splits * self.n_repeats) \
			)
		self.mdlscorer = make_scorer( \
			self.ScoreMetric, \
			greater_is_better=True, \
			needs_proba=True, \
			needs_threshold=False
		)

	def _DefineModel(self):
		'''
		This method initiliazes models for the analysis
        It also initializes the CV methods and class-weights that could be tuned going ahead.
		:return:
		'''
		# Commonly used CV strategies for later usage:
		self.all_cv = {
			'KF': KFold(n_splits=self.n_splits, shuffle=True, random_state=self.state),
			'RKF': RKF(n_splits=self.n_splits, n_repeats=self.n_repeats, random_state=self.state),
			# TODO 5-30
			'RSKF': RSKF(n_splits=self.n_splits, n_repeats=self.n_repeats, random_state=self.state),
			'SKF': SKF(n_splits=self.n_splits, shuffle=True, random_state=self.state),
			'SGKF': SGKF(n_splits=self.n_splits, shuffle=True, random_state=self.state)
		}
		self.Mdl_Master = {
			'ada': ensemble.AdaBoostClassifier(),
			# 'XGB1C': XGBC(
			# 	**{
			# 		'tree_method': 'hist',
			# 		'device': 'cuda' if self.gpu_switch == 'ON' else 'cpu',
			# 		'objective': 'binary:logistic',
			# 		'eval_metric': 'auc',
			# 		'random_state': self.state,
			# 		'colsample_bytree': 0.25,  # 构建弱学习器时，对特征随机采样的比例，默认值为1。
			# 		'learning_rate': 0.07,
			# 		'max_depth': 8,
			# 		'n_estimcator': 1100,
			# 		'reg_alpha': 0.7,
			# 		'reg_lambda': 0.7,
			# 		'min_child_weight': 22,  # 指定孩子节点中最小的样本权重和，
			# 		# 如果一个叶子节点的样本权重和小于min_child_weight则拆分过程结束，默认值为1。
			# 		'early_stopping_rounds': self.nbrnd_erly_stp,  # 指定迭代多少次没有得到优化则停止训练，
			# 		# 默认值为None，表示不提前停止训练。如果设置了此参数，则模型会生成三个属性：
			# 		# best_score, best_iteration, best_ntree_limit
			# 		# 注意：evals 必须非空才能生效，如果有多个数据集，则以最后一个数据集为准。
			# 		'verbosity': 0,  # 训练中是否打印每次训练的结果
			# 		# 开启参数verbosity，在数据巨大，预料到算法运行会非常缓慢的时候
			# 		# 可以使用这个参数来监控模型的训练进度
			# 		'enable_categorical': True
			# 	}),
			# 'XGB2C': XGBC(**{
			# 	'tree_method': 'hist',
			# 	'device': 'cuda' if self.gpu_switch == 'ON' else 'cpu',
			# 	'eval_metric': 'auc',
			# 	'random_state': self.state,
			# 	'colsample_bytree': 0.4,
			# 	'learning_rate': 0.06,
			# 	'max_depth': 9,
			# 	'n_estimator': 2500,
			# 	'reg_alpha': 0.12,
			# 	'reg_lambda': 0.8,
			# 	'min_child_weight': 15,
			# 	'early_stopping_rounds': self.nbrnd_erly_stp,
			# 	'verbosity': 0,
			# 	'enable_categorical': True,
			# }),
			# 'XGB3C': XGBC(**{
			# 	'tree_method': 'hist',
			# 	'device': 'cuda' if self.gpu_switch == 'ON' else 'cpu',
			# 	'objective': 'binary:logistic',
			# 	'eval_metric': 'auc',
			# 	'random_state': self.state,
			# 	'colsample_bytree': 0.5,
			# 	'learning_rate': 0.055,
			# 	'max_depth': 9,
			# 	'n_estimator': 3000,
			# 	'reg_alpha': .2,
			# 	'reg_lambda': 0.6,
			# 	'min_child_weight': 25,
			# 	'early_stopping_rounds': CFG.nbrnd_erly_stp,
			# 	'verbosity': 0,
			# 	'enable_categorical': True,
			# }),
			# 'XGB4C': XGBC(**{
			# 	'tree_method': 'hist',
			# 	'device': 'cuda' if self.gpu_switch == 'ON' else 'cpu',
			# 	'eval_metric': 'auc',
			# 	'random_state': self.state,
			# 	'colsample_bytree': 0.8,
			# 	'learning_rate': 0.082,
			# 	'max_depth': 7,
			# 	'n_estimator': 2000,
			# 	'reg_alpha': 0.005,
			# 	'reg_lambda': 0.95,
			# 	'min_child_weight': 26,
			# 	'early_stopping_round': self.nbrnd_erly_stp,
			# 	'verbosity': 0,
			# 	'enable_categorical': True
			# }),
			# 'LGBM1C': LGBMC(**{
			# 	'device': 'gpu' if self.gpu_switch == 'ON' else 'cpu',
			# 	'objective': 'binary',
			# 	'boosting_type': 'gbdt',  # 用于指定弱学习器的类型，默认值为 ‘gbdt’，
			# 	# 表示使用基于树的模型进行计算。
			# 	# 还可以选择为 ‘gblinear’ 表示使用线性模型作为弱学习器。
			# 	'metric': 'auc',  # 用于指定评估指标，可以传递各种评估方法组成的list。
			# 	'random_state': self.state,
			# 	'colsample_bytree': 0.56,  # 构建弱学习器时，对特征随机采样的比例，默认值为1。
			# 	'subsample': 0.35,  # 默认值1，指定采样出 subsample * n_samples 个样本用于训练弱学习器。
			# 	# 注意这里的子采样和随机森林不一样，随机森林使用的是放回抽样，而这里是不放回抽样。
			# 	# 取值在(0, 1)之间，
			# 	# 设置为1表示使用所有数据训练弱学习器。如果取值小于1，
			# 	# 则只有一部分样本会去做GBDT的决策树拟合。选择小于1的比例可以减少方差，即防止过拟合，
			# 	# 但是会增加样本拟合的偏差，因此取值不能太低。
			# 	'learning_rate': 0.05,
			# 	'max_depth': 6,
			# 	'n_estimators': 3000,
			# 	'num_leaves': 140,
			# 	'reg_alpha': 0.14,  # 正则化参数 L1正则化系数
			# 	'reg_lambda': 0.85,  # L2正则化系数
			# 	'verbosity': -1,
			# 	'categorical_feature': [f'name:{c}' for c in self.cat_cols]  # 指定哪些是类别特征。
			# }),
			# 'LGBM2C': LGBMC(**{'device': "gpu" if self.gpu_switch == "ON" else "cpu",
			# 				   'objective': 'binary',
			# 				   'boosting_type': 'gbdt',
			# 				   'data_sample_strategy': "goss",
			# 				   'metric': "auc",
			# 				   'random_state': self.state,
			# 				   'colsample_bytree': 0.20,
			# 				   'subsample': 0.25,
			# 				   'learning_rate': 0.10,
			# 				   'max_depth': 7,
			# 				   'n_estimators': 3000,
			# 				   'num_leaves': 120,
			# 				   'reg_alpha': 0.15,
			# 				   'reg_lambda': 0.90,
			# 				   'verbosity': -1,
			# 				   'categorical_feature': [f"name: {c}" for c in self.cat_cols],
			# 				   }
			# 				),
			#
			# 'LGBM3C': LGBMC(**{'device': "gpu" if CFG.gpu_switch == "ON" else "cpu",
			# 				   'objective': 'binary',
			# 				   'boosting_type': 'gbdt',
			# 				   'metric': "auc",
			# 				   'random_state': self.state,
			# 				   'colsample_bytree': 0.45,
			# 				   'subsample': 0.45,
			# 				   'learning_rate': 0.06,
			# 				   'max_depth': 6,
			# 				   'n_estimators': 3000,
			# 				   'num_leaves': 125,
			# 				   'reg_alpha': 0.05,
			# 				   'reg_lambda': 0.95,
			# 				   'verbosity': -1,
			# 				   'categorical_feature': [f"name: {c}" for c in self.cat_cols],
			# 				   }
			# 				),
			#
			# 'LGBM4C': LGBMC(**{'device': "gpu" if self.gpu_switch == "ON" else "cpu",
			# 				   'objective': 'binary',
			# 				   'boosting_type': 'gbdt',
			# 				   'metric': "auc",
			# 				   'random_state': self.state,
			# 				   'colsample_bytree': 0.55,
			# 				   'subsample': 0.55,
			# 				   'learning_rate': 0.085,
			# 				   'max_depth': 7,
			# 				   'n_estimators': 3000,
			# 				   'num_leaves': 105,
			# 				   'reg_alpha': 0.08,
			# 				   'reg_lambda': 0.995,
			# 				   'verbosity': -1,
			# 				   }
			# 				),
			# 'CB1C': CBC(**{
			# 	'task_type': 'GPU' if self.gpu_switch == 'ON' else 'CPU',
			# 	'objective': 'Logloss',
			# 	'eval_metric': 'AUC',
			# 	'bagging_temperature': 0.1,  # 贝叶斯套袋控制强度，区间[0, 1]。默认1
			# 	'colsample_bylevel': 0.88,
			# 	'iterations': 3000,  # 最大树数。默认1000。
			# 	'learning_rate': 0.065,
			# 	'od_wait': 12,  # 与early_stopping_rounds部分相似，
			# 	# od_wait为达到最佳评估值后继续迭代的次数，
			# 	# 检测器为IncToDec时达到最佳评估值后继续迭代n次（n为od_wait参数值）；
			# 	# 检测器为Iter时达到最优评估值后停止，默认值20`
			# 	'max_depth': 7,
			# 	'l2_leaf_reg': 1.75,  # l2正则项，别名：reg_lambda
			# 	'min_data_in_leaf': 25,  # 叶子结点最小样本量
			# 	'random_strength': 0.1,  # 设置特征分裂信息增益的扰过拟合。
			# 	# 子树分裂时，正常会寻找最大信息增益的特征+分裂点进行分裂，
			# 	# 此处对每个特征+分裂点的信息增益值+扰动项后再确定最大值。扰动项服从正态分布、均值为0，
			# 	# random_strength参数值会作为正态分布的方差，默认值1、对应标准正态分布；设置0时则无扰动项
			# 	'max_bin': 100,  # 数值型特征的分箱数，别名max_bin，取值范围[1,65535]、默认值254（CPU下)
			# 	'verbose': 0,  # # 模型训练过程的信息输出等级，取值Silent（不输出信息）、
			# 	# Verbose（默认值，输出评估指标、已训练时间、剩余时间等）、
			# 	# Info（输出额外信息、树的棵树）、Debug（debug信息）
			# 	'use_best_model': True  # 让模型使用效果最优的子树棵树/迭代次数，
			# 	# 使用验证集的最优效果对应的迭代次数（eval_metric：评估指标，eval_set：验证集数据），
			# 	# 布尔类型可取值0，1（取1时要求设置验证集数据）
			# }),
			# "CB2C": CBC(**{'task_type': "GPU" if self.gpu_switch == "ON" else "CPU",
			# 			   'objective': 'Logloss',
			# 			   'eval_metric': "AUC",
			# 			   'bagging_temperature': 0.5,
			# 			   'colsample_bylevel': 0.50,
			# 			   'iterations': 2500,
			# 			   'learning_rate': 0.04,
			# 			   'od_wait': 24,
			# 			   'max_depth': 8,
			# 			   'l2_leaf_reg': 1.235,
			# 			   'min_data_in_leaf': 25,
			# 			   'random_strength': 0.35,
			# 			   'max_bin': 160,
			# 			   'verbose': 0,
			# 			   'use_best_model': True,
			# 			   }
			# 			),
			# "CB3C": CBC(**{'task_type': "GPU" if self.gpu_switch == "ON" else "CPU",
			# 			   'objective': 'Logloss',
			# 			   'eval_metric': "AUC",
			# 			   'bagging_temperature': 0.2,
			# 			   'colsample_bylevel': 0.85,
			# 			   'iterations': 2500,
			# 			   'learning_rate': 0.025,
			# 			   'od_wait': 10,
			# 			   'max_depth': 7,
			# 			   'l2_leaf_reg': 1.235,
			# 			   'min_data_in_leaf': 8,
			# 			   'random_strength': 0.60,
			# 			   'max_bin': 160,
			# 			   'verbose': 0,
			# 			   'use_best_model': True,
			# 			   }
			# 			),
			# "CB4C": CBC(**{'task_type': "GPU" if self.gpu_switch == "ON" else "CPU",
			# 			   'objective': 'Logloss',
			# 			   'eval_metric': "AUC",
			# 			   'grow_policy': 'Lossguide',
			# 			   'colsample_bylevel': 0.25,
			# 			   'iterations': 2500,
			# 			   'learning_rate': 0.035,
			# 			   'od_wait': 24,
			# 			   'max_depth': 7,
			# 			   'l2_leaf_reg': 1.80,
			# 			   'random_strength': 0.60,
			# 			   'max_bin': 160,
			# 			   'verbose': 0,
			# 			   'use_best_model': True,
			# 			   }
			# 			),
			# 'HGB1C': HGBC(
			# 	loss='log_loss',
			# 	learning_rate=0.06,
			# 	max_iter=800,  # boosting过程的最大迭代次数，即二分类的最大树数。对于多类分类，每次迭代都会构建n_classes 树。
			# 	max_depth=6,  # 每棵树的最大深度。树的深度是从根到最深叶的边数。默认情况下，深度不受限制。。
			# 	min_samples_leaf=12,  # 每片叶子的最小样本数。对于少于几百个样本的小型数据集，
			# 	# 建议降低此值，因为只会构建非常浅的树。
			# 	l2_regularization=1.15,  # L2 正则化参数。使用 0 表示不进行正则化。
			# 	validation_fraction=0.1,  # 留出作为提前停止验证数据的训练数据的比例(或绝对大小)。
			# 	# 如果没有，则对训练数据进行提前停止。仅在执行提前停止时使用。
			# 	n_iter_no_change=self.nbrnd_erly_stp,  # 用于确定何时“early stop”。
			# 	# 当最后一个 n_iter_no_change 分数都没有优于 n_iter_no_change - 1 -th-to-last 分数时，
			# 	# 拟合过程将停止，达到一定的容差。仅在执行提前停止时使用
			# 	random_state=self.state
			# ),
			# 'HGB2C': HGBC(
			# 	loss='log_loss',
			# 	learning_rate=0.035,
			# 	max_iter=700,
			# 	max_depth=7,
			# 	min_samples_leaf=9,
			# 	l2_regularization=1.75,
			# 	validation_fraction=0.1,
			# 	n_iter_no_change=self.nbrnd_erly_stp,
			# 	random_state=self.state
			# ),
		}
		return self

	def ScoreMetric(self, ytrue, ypred):
		'''
		This is the metric function for the competition scoring
		:param ytrue:
		:param y_pred:
		:return:
		'''
		return accuracy_score(ytrue, ypred)

	def ClbMetric(self, ytrue, ypred):
		'''
		This is the calibration metric
		:param ytrue:
		:param ypred:
		:return:
		'''
		return brier_score_loss(ytrue, ypred)

	def PostProcessPred(self, ypred):
		'''
		This is an optional post-processing method
		:param ypred:
		:return:
		'''
		return np.clip(ypred, a_min=0, a_max=1)

	def TrainMdl(self, target: str, test_preds_req: str = 'Y', save_models='N'):
		'''
		This method trains and infers from the model suite and
		returns the predictions and scores
        It optionally predicts the test set too, if desired by the user
        Source for dropped columns:-
        https://www.kaggle.com/competitions/playground-series-s4e3/discussion/482401
		:param target:
		:param test_preds_req:
		:param save_models:
		:return:
		'''
		# Initializing I-O :-
		X, y, Xt = self.Xtrain[self.sel_cols], \
				   self.ytrain.copy(), \
				   self.Xtest[self.sel_cols]
		cols_drop = [
			'Source', 'id', 'Sum_of_Luminosity', \
			'X_Perimeter', 'SigmoidOfAreas', \
			'Edges_X_Index', 'Y_Minimum', 'Y_Maximum'
		]
		ens = OptunaEnsembler()
		self.FtreImp = pd.DataFrame(
			columns=self.methods, \
			index=[c for c in self.sel_cols if c not in cols_drop] \
			).fillna(0)

		# Making CV folds:-
		for fold_nb, (train_idx, dev_idx) in tqdm(enumerate(self.cv.split(X, self.y_grp))):
			Xtr = X.iloc[train_idx].drop(columns=cols_drop, errors='ignore')
			Xdev = X.iloc[dev_idx].drop(columns=cols_drop, errors='ignore')
			ytr = y.loc[y.index.isin(Xtr.index)]
			ydev = y.loc[y.index.isin(Xdev.index)]
			# Initializing the OOF and test set predictions:-
			oof_preds = pd.DataFrame(columns=self.methods, index=Xdev.index)
			mdl_preds = pd.DataFrame(columns=self.methods, index=Xt.index)
			# PrintColor(f"\n{' = ' * 5} Fold {fold_nb + 1} {' = ' * 5}\n")
			# Initializing models across methods:-
			for method in tqdm(self.methods[0:1]):
				print(f'{datetime.now()}; fold: { fold_nb} ; method :{method}  start ')
				model = Pipeline(steps=[('M', self.Mdl_Master.get(method))])
				# Fitting the model:-
				if 'CB' in method:
					model.fit(Xtr, ytr, \
							  M__eval_set=[(Xdev, ydev)], \
							  M__verbose=0,
							  M__early_stopping_rounds=CFG.nbrnd_erly_stp)
				elif 'LGBM' in method:
					model.fit(Xtr, ytr, \
							  M__eval_set=[(Xdev, ydev)], \
							  M__callbacks=[log_evaluation(0), \
											early_stopping(stopping_rounds=CFG.nbrnd_erly_stp, verbose=False, ),
											]
							  )
				elif 'XGB' in method:
					model.fit(Xtr, ytr, \
							  M__eval_set=[(Xdev, ydev)], \
							  M__verbose=0, )
				else:
					model.fit(Xtr, ytr)
				# Collating feature importance:-
				try:
					self.FtreImp[method] += model['M'].feature_importances_
				except:
					pass
				# Collecting predictions and scores and post-processing OOF based on model method:-
				dev_preds = model.predict(Xdev)
				# dev_preds = model.predict_proba(Xdev)[:, 1]
				# train_preds = model.predict_proba(Xtr)[:, 1]
				train_preds = model.predict(Xtr)
				tr_score = self.ScoreMetric(ytr.values.flatten(), \
											train_preds)
				score = self.ScoreMetric(ydev.values.flatten(), dev_preds)
				# PrintColor(f'OOF={score:.5f} | train = {tr_score:.5f} | {method}', color=Fore.CYAN)
				oof_preds[method] = dev_preds  # TODO importance
				# Integrating the predictions and scores:-
				self.Scores.at[fold_nb, method] = np.round(score, decimals=6)
				self.TrainScores.at[fold_nb, method] = np.round(tr_score, decimals=6)
				if test_preds_req == 'Y':
					mdl_preds[method] = self.PostProcessPred(model.predict_proba(Xt.drop(columns=cols_drop,
																						 errors='ignore')))
				print(f'{datetime.now()}; fold: { fold_nb} ; method :{method}  end ')
			try:
				del dev_preds, train_preds, tr_score, score
			except:
				pass
			# Ensembling the predictions:-
			oof_preds['Ensemble'] = ens.fit_predict(ydev, oof_preds[self.methods])
			score = self.ScoreMetric(ydev, oof_preds['Ensemble'].values)
			self.OOF_Preds = pd.concat([self.OOF_Preds, oof_preds], axis=0, ignore_index=False)
			self.Scores.at[fold_nb, 'Ensemble'] = np.round(score, 6)
			if test_preds_req == 'Y':
				mdl_preds['Ensemble'] = ens.predict(mdl_preds[method])
				self.Mdl_Preds = pd.concat([self.Mdl_Preds, mdl_preds], axis=1, ignore_index=False)
		# Averaging the predictions afeter all folds:-
		self.OOF_Preds = self.OOF_Preds.groupby(level=0).mean()
		if test_preds_req == 'Y':
			self.Mdl_Preds = self.Mdl_Preds[self.methods + ['Ensemble']].groupby(level=0).mean()
		return self.OOF_Preds['Ensemble'].astype(np.int8), self.Mdl_Preds, self.Scores, self.TrainScores

	def MakePseudoLbl(self, up_cutoff: float, low_cutoff: float, **kwargs):
		"""
		This method makes pseudo-labels using confident test set predictions to add to the training data
		:param up_cutoff:
		:param low_cutoff:
		:param kwargs:
		:return:
		"""
		df = self.Mdl_Preds.loc[ \
			(self.Mdl_Preds.Ensemble >= up_cutoff) | (self.Mdl_Preds.Ensemble <= low_cutoff), \
			'Ensemble'
		]
		# PrintColor(f'-->Pseudo Label additions form test set={df.shape[0]:,.0f}', color=Fore.RED)
		df = df.astype(np.uint8)
		new_ytrain = pd.concat([self.ytrain, df], axis=0, ignore_index=True)
		new_ytrain.index = range(len(new_ytrain))

		new_Xtrain = pd.concat([self.Xtrain, self.Xtest.loc[df.index]], \
							   axis=0, \
							   ignore_index=True)
		new_Xtrain.index = range(len(new_Xtrain))
		#  Verifying the additions:-
		# PrintColor(f"---> Revised train set shapes after pseudo labels = {new_Xtrain.shape} {new_ytrain.shape}");
		return new_Xtrain, new_ytrain

	def MakeMLPlots(self):
		'''
		This method makes plots for the ML models,
		 including feature importance and calibration curves
		:return:
		'''
		pass


from sklearn.preprocessing import LabelEncoder

# pp = Preprocessor();
# pp.DoPreprocessing();
target = 'Target'
cols_drop = ['id', 'Target']
ytrain = train[CFG.targets]
label = LabelEncoder()
ytrain = label.fit_transform(ytrain)
ytrain = pd.DataFrame(ytrain).astype(np.int8)

train = train.drop(cols_drop, axis=1)
sel_cols = train.columns
print('2' * 100)
print(train.nunique())
featureCount = train.nunique()
cat_ftre = featureCount.loc[featureCount <= 10].index.to_list()
cont_cols = [c for c in featureCount.index if c not in cat_ftre]

if CFG.ML == 'Y':
	OOF_Preds, Mdl_Preds, Scores, TrainScores = pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
	md = MdlDeveloper(train, ytrain, ytrain, test, sel_cols=sel_cols, cat_cols=cat_ftre, enc_cols=[])
	oof_preds, mdl_preds, scores, trainscores = md.TrainMdl(test_preds_req='Y', target=target)
	print(oof_preds)
	# OOF_Preds = pd.concat([oof_preds.assign(Target=target), OOF_Preds], \
	# 					  axis=0, \
	# 					  ignore_index=False)
	OOF_Preds=pd.DataFrame({"id":test.id,"target":OOF_Preds})
	Mdl_Preds = pd.concat([mdl_preds.assign(Target=target), Mdl_Preds], \
						  axis=0, \
						  ignore_index=False)
	Scores = pd.concat([scores.assign(Target=target), Scores], \
					   axis=0, \
					   ignore_index=True)
	TrainScores = pd.concat([trainscores.assign(Target=target), TrainScores], \
							axis=0, \
							ignore_index=True)
	sub_f1=pd.DataFrame({'id':test['id'],'target':Mdl_Preds})
	sub_f1.to_csv(f'Submission_V{CFG.version_nb}.csv', index=False)

# if CFG.ML == 'Y':
	# sub_f1=pd.DataFrame({'id':test['id'],'target':MDL_Preds })
	# for col in CFG.targets:
	# sub_f1 = 1 - Mdl_Preds.loc[Mdl_Preds.Target == 'Target', 'Ensemble']
	# sub1 = pd.read_csv(f'../input/playgrounds4e03ancillary/89652_submission.csv')[CFG.targets]
	# pp.sub_f1[CFG.targets] = pp.sub_f1[CFG.targets].values * 0.1 + sub1 * 0.9
	#
	# pp.sub_f1.to_csv(f'Submission_V{CFG.version_nb}.csv', index=False)
	# sub_f1.to_csv(f'Submission_V{CFG.version_nb}.csv', index=False)
# OOF_Preds.to_csv(f'OOF_Preds_V{CFG.version_nb}.csv', index=False)
# Mdl_Preds.to_csv(f'Mdl_Preds_V{CFG.version_nb}.csv', index=False)

# from flaml import AutoML
#
# automl = AutoML()
# y = train.pop('Target')
# X = train
# print(f'{datetime.now()} automl start !')
# automl.fit(X, y, task='classification', metric='roc_auc_ovo', time_budget=3600 * 3)
# print(f'{datetime.now()} automl end !')
#
# y_pred = automl.predict(test)
# print('y_pred[:5]:')
# print(y_pred[:5])
# df = pd.DataFrame(y_pred, columns=['Target'])
# print('df.head():')
# print(df.head())
#
# sol = pd.read_csv('../input/playground-series-s4e6/sample_submission.csv')
# print('sol.head():')
# print(sol.head())
# sol.to_csv('./roc_auc_ovo.csv', index=False)
