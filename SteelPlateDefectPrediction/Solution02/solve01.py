# -*- coding: utf-8 -*-
# @Time : 2024/4/5 16:04
# @Author : nanji
# @Site :
# https://www.kaggle.com/code/arunklenin/ps4e3-steel-plate-fault-prediction-multilabel
# @File : solve01.py
# @Software: PyCharm 
# @Comment : PS4E3 | Steel Plate Fault Prediction |Multilabel
import sklearn
import numpy as np
import os
import datetime
import pandas as pd
import matplotlib.pyplot as plt
import missingno as msno
from sklearn.datasets import load_boston
import pandas as pd
import numpy as np
import warnings
from prettytable import PrettyTable
import seaborn as sns

sns.set(style='darkgrid', font_scale=1.4)
from tqdm import tqdm
from tqdm.notebook import tqdm as tqdm_notebook

tqdm_notebook.get_lock().locks = []
import concurrent.futures
from copy import deepcopy

from functools import partial
from itertools import combinations
from sklearn.feature_selection import f_classif

from sklearn.preprocessing import LabelEncoder, StandardScaler, \
	MinMaxScaler, PowerTransformer, FunctionTransformer
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from itertools import combinations
from sklearn.impute import SimpleImputer
import xgboost as xg
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, \
	mean_squared_log_error, \
	roc_auc_score, \
	accuracy_score, \
	f1_score, \
	precision_recall_curve, \
	log_loss
from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer
from gap_statistic.optimalK import OptimalK
from scipy import stats
import statsmodels.api as sm

from scipy.stats import ttest_ind
from scipy.stats import boxcox

import math
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.base import BaseEstimator, TransformerMixin
import optuna
import cmaes
# pip install cmaes
import xgboost as xgb
import lightgbm as lgb
from category_encoders import OneHotEncoder, OrdinalEncoder, CountEncoder, CatBoostEncoder
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.ensemble import RandomForestClassifier, \
	HistGradientBoostingClassifier, \
	GradientBoostingClassifier, \
	ExtraTreesClassifier, \
	AdaBoostClassifier

from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from catboost import CatBoost, CatBoostRegressor, CatBoostClassifier
from sklearn.svm import NuSVC, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.impute import KNNImputer
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from catboost import Pool
import re
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD
# Suppress warnings
import warnings

warnings.filterwarnings('ignore')
import pandas as pd

# 666666
pd.pandas.set_option('display.max_columns', None)

train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
original = pd.read_csv('../input/SteelPlatesFaults.csv')
print(train.columns)
print('0' * 100)
print(train.info())
train.drop(columns=['id'], inplace=True)
test.drop(columns=['id'], inplace=True)
train_copy = train.copy()
test_copy = test.copy()
original_copy = original.copy()
print('1' * 100)
print(original.shape)
device = 'cpu'
train = pd.concat([train, original], axis=0)
train.reset_index(inplace=True, drop=True)
target = ['Pastry', 'Z_Scratch', 'K_Scatch', 'Stains', 'Dirtiness', 'Bumps', 'Other_Faults']

# print(original.head())
cont_cols = test.columns
colors = ['blue', 'orange', 'green']
num_plots = len(cont_cols)
num_cols = 3
num_rows = -(-num_plots // num_cols)


# fig, axes = plt.subplots(num_rows, num_cols, \
# 						 figsize=(21, 5 * num_rows))
# Adjust the figure size as needed
# for i, feature in enumerate(cont_cols):
# 	row = i // num_cols
# 	col = i % num_cols
# 	ax = axes[row, col] if num_rows > 1 else axes[col]
# 	sns.histplot(train_copy[feature], kde=True, \
# 				 color=colors[0], \
# 				 label='Train', \
# 				 alpha=0.5, \
# 				 bins=30, \
# 				 ax=ax)
# 	sns.histplot(test_copy[feature], kde=True, \
# 				 color=colors[1], \
# 				 label='Test', \
# 				 alpha=0.5, \
# 				 bins=30, \
# 				 ax=ax)
# 	sns.histplot(original[feature], kde=True, \
# 				 color=colors[2], \
# 				 label='Original', alpha=.5, bins=30, ax=ax)
# 	ax.set_title(f'Distribution of {feature}')
# 	ax.set_xlabel(feature)
# 	ax.set_ylabel('Frequency')
# 	ax.legend()
# if num_cols % num_cols != 0:
# 	for j in range(num_plots % num_cols, num_cols):
# 		axes[-1, j].axis('off')
# plt.tight_layout()
# plt.show()


def OHE(train_df, test_df, cols, target):
	'''
	Function for one hot encoding, it first combined the data so that no category is missed and
	the category with least frequency can be dropped because of redunancy
	'''
	combined = pd.concat([train_df, test_df], axis=0)
	for col in cols:
		one_hot = pd.get_dummies(combined[col]).astype(int)
		counts = combined[col].value_counts()
		min_count_category = counts.idxmin()
		one_hot = one_hot.drop(min_count_category, axis=1)
		one_hot.columns = [str(f) + col for f in one_hot.columns]
		combined = pd.concat([combined, one_hot], axis='columns')
		combined = combined.loc[:, ~combined.columns.duplicated()]
	train_ohe = combined[:len(train_df)]
	test_ohe = combined[len(train_df):]
	test_ohe.reset_index(inplace=True, drop=True)
	test_ohe.drop(columns=[target], inplace=True)
	return train_ohe, test_ohe


# TODO new category
cat_cols = [f for f in test.columns if test[f].nunique() / test.shape[0] * 100 < 5 and test[f].nunique() > 2]
print(test[cat_cols].nunique())


# def nearst_val(target):
# 	return min(common, key=lambda x: abs(x - target))


global cat_cols_updated
cat_cols_updated = []
for col in cat_cols:
	train[f'{col}_cat'] = train[col]
	test[f'{col}_cat'] = test[col]
	cat_cols_updated.append(f'{col}_cat')
	uncommon = list(
		(set(test[col].unique()) | set(train[col].unique())) - (set(test[col].unique()) & set(train[col].unique())))
	# if uncommon:
	# 	common = list(set(test[col].unique()) & set(train[col].unique()))
		# train[f'{col}_cat'] = train[col].apply(nearst_val)
		# test[f'{col}_cat'] = test[col].apply(nearst_val)

import os

train_path = '../input/train_cat.csv'
test_path = '../input/test_cat.csv'
if not os.path.exists(train_path):
	train.to_csv(train_path)
	test.to_csv(test_path)
train = pd.read_csv(train_path)
test = pd.read_csv(test_path)
print('0' * 100)


def high_freq_ohe(train, test, extra_cols, target, n_limit=50):
	'''
	  If you wish to apply one hot encoding on a feature with so many unique values, then this can be applied,
	  where it takes a maximum of n categories and drops the rest of them treating as rare categories
	  '''
	train_copy = train.copy()
	test_copy = test.copy()
	ohe_cols = []
	for col in extra_cols:
		dict1 = train_copy[col].value_counts().to_dict()
		orderd = dict(sorted(dict1.items(), key=lambda x: x[1], reverse=True))
		rare_keys = list([*orderd.keys()][n_limit:])
		# ext_keys = [f[0] for f in orderd.items() if f[1] < 50]
		rare_key_map = dict(zip(rare_keys, np.full(len(rare_keys), 9999)))

		train_copy[col] = train_copy[col].replace(rare_key_map)
		test_copy[col] = test_copy[col].replace(rare_key_map)
	train_copy, test_copy = OHE(train_copy, test_copy, extra_cols, target)
	drop_cols = [f for f in train_copy.columns if '9999' in f or train_copy[f].nunique() == 1]
	train_copy = train_copy.drop(columns=drop_cols)
	test_copy = test_copy.drop(columns=drop_cols)
	return train_copy, test_copy


def cat_encoding(train, test, target):
	global overall_best_score
	global overall_best_col
	table = PrettyTable()
	table.field_names = ['Feature', 'Encoded Features', 'Log Loss Score']
	train_copy = train.copy()
	test_copy = test.copy()
	train_dum = train.copy()
	for feature in cat_cols_updated:
		#         print(feature)
		#         cat_labels = train_dum.groupby([feature])[target].mean().sort_values().index
		#         cat_labels2 = {k: i for i, k in enumerate(cat_labels, 0)}
		#         train_copy[feature + "_target"] = train[feature].map(cat_labels2)
		#         test_copy[feature + "_target"] = test[feature].map(cat_labels2)
		dic = train[feature].value_counts().to_dict()
		train_copy[feature + '_count'] = train[feature].map(dic)
		test_copy[feature + '_count'] = test[feature].map(dic)
		dic2 = train[feature].value_counts().to_dict()
		list1 = np.arange(len(dic2.values()))
		dic3 = dict(zip(list(dic2.keys()), list1))
		train_copy[feature + '_count_label'] = train[feature].replace(dic3).astype(float)
		test_copy[feature + '_count_label'] = test[feature].replace(dic3).astype(float)

		temp_cols = [feature + '_count', feature + '_count_label']
		if train_copy[feature].nunique() <= 5:
			train_copy[feature] = train_copy[feature].astype(str) + '_' + feature
			test_copy[feature] = test_copy[feature].astype(str) + '_' + feature
			train_copy, test_copy = OHE(train_copy, test_copy, [feature], target)
		else:
			train_copy, test_copy = high_freq_ohe(train_copy, test_copy, [feature], target, n_limit=5)
		train_copy = train_copy.drop(columns=[feature])
		test_copy = test_copy.drop(columns=[feature])
		kf = KFold(n_splits=5, shuffle=True, random_state=42)
		auc_scores = []
		for f in temp_cols:
			X = train_copy[[f]].values
			y = train_copy[target].astype(int).values
			auc = []
			for train_idx, val_idx in kf.split(X, y):
				X_train, y_train = X[train_idx], y[train_idx]
				x_val, y_val = X[val_idx], y[val_idx]
				model = HistGradientBoostingClassifier(max_iter=300, \
													   learning_rate=0.02, \
													   max_depth=6, \
													   random_state=42 )
				model.fit(X_train,y_train)
				y_pred = model.predict_proba(x_val)[:, 1]
				auc.append(roc_auc_score(y_val,y_pred))
			auc_scores.append(auc)

		best_col, best_auc = sorted(auc_scores, key=lambda x: x[1], reverse=True)[0]
		corr = train_copy[temp_cols].corr(method='pearson')
		corr_with_best_col = corr[best_col]
		cols_to_drop = [f for f in temp_cols if corr_with_best_col[f] > 0.5
						and f != best_col]
		final_selection = [f for f in temp_cols if f not in cols_to_drop]
		if cols_to_drop:
			train_copy = train_copy.drop(columns=cols_to_drop)
			test_copy = test_copy.drop(columns=cols_to_drop)
		table.add_row([feature, best_col, best_auc])
	return train_copy, test_copy


class Splitter:
	def __init__(self, test_size=0.2, kfold=True, n_splits=5):
		self.test_size = test_size
		self.kfold = kfold
		self.n_splits = n_splits

	def split_data(self, X, y, random_state_list):
		if self.kfold:
			for random_state in random_state_list:
				kf = KFold(n_splits=self.n_splits, random_state=random_state, shuffle=True)
				for train_index, val_index in kf.split(X, y):
					X_train, X_val = X.iloc[train_index], X.iloc[val_index]
					y_train, y_val = y.iloc[train_index], y.iloc[val_index]
					yield X_train, X_val, y_train, y_val


class Classifier:
	def __init__(self, n_estimators=100, device='cpu', random_state=0):
		self.n_estimators = n_estimators
		self.device = device
		self.random_state = random_state
		self.models = self._define_model()
		self.len_models = len(self.models)

	def _define_model(self):
		xgb_params = {
			'n_estimators': self.n_estimators,
			'learning_rate': 0.1,
			'max_depth': 4,
			'subsample': 0.8,
			'colsample_bytree': 0.1,
			'n_jobs': -1,
			'eval_metric': 'logloss',
			'objective': 'binary:logistic',
			'tree_method': 'hist',
			'verbosity': 0,
			'random_state': self.random_state,
			#             'class_weight':class_weights_dict,
		}
		if self.device == 'gpu':
			xgb_params['tree_method'] = 'gpu_hist'
			xgb_params['predictor'] = 'gpu_predictor'
		xgb_params2 = xgb_params.copy()
		xgb_params2['subsample'] = 0.5
		xgb_params2['max_depth'] = 9
		xgb_params2['learning_rate'] = 0.045
		xgb_params2['colsample_bytree'] = 0.3

		xgb_params3 = xgb_params.copy()
		xgb_params3['subsample'] = 0.6
		xgb_params3['max_depth'] = 6
		xgb_params3['learning_rate'] = 0.02
		xgb_params3['colsample_bytree'] = 0.7

		xgb_params4 = xgb_params.copy()
		xgb_params4['subsample'] = 0.5943421542786502
		xgb_params4['max_depth'] = 6
		xgb_params4['learning_rate'] = 0.109
		xgb_params4['colsample_bytree'] = 0.5595039093313848
		lgb_params = {
			'n_estimators': self.n_estimators,
			'max_depth': 8,
			'learning_rate': 0.02,
			'subsample': 0.20,
			'colsample_bytree': 0.56,
			'reg_alpha': 0.25,
			'reg_lambda': 5e-08,
			'objective': 'binary',
			'boosting_type': 'gbdt',
			'device': self.device,
			'random_state': self.random_state,
			'verbose': -1,
			#             'class_weight':class_weights_dict,
		}
		lgb_params2 = {
			'n_estimators': self.n_estimators,
			'max_depth': 5,
			'learning_rate': 0.015,
			'subsample': 0.50,
			'colsample_bytree': 0.1,
			'reg_alpha': 0.07608657669988828,
			'reg_lambda': 0.2255036530113883,
			'objective': 'binary',
			'boosting_type': 'gbdt',
			'device': self.device,
			'random_state': self.random_state,
		}
		lgb_params3 = lgb_params.copy()
		lgb_params3['subsample'] = 0.9
		lgb_params3['reg_lambda'] = 0.3461495211744402
		lgb_params3['reg_alpha'] = 0.3095626288582237
		lgb_params3['max_depth'] = 8
		lgb_params3['learning_rate'] = 0.007
		lgb_params3['colsample_bytree'] = 0.5

		lgb_params4 = lgb_params2.copy()
		lgb_params4['subsample'] = 0.3
		lgb_params4['reg_lambda'] = 0.49406951573373614
		lgb_params4['reg_alpha'] = 0.16269100796945424
		lgb_params4['max_depth'] = 9
		lgb_params4['learning_rate'] = 0.117
		lgb_params4['colsample_bytree'] = 0.3
		cb_params = {
			'iterations': self.n_estimators,
			'depth': 13,
			'learning_rate': 0.015,
			'l2_leaf_reg': 0.5,
			'random_strength': 0.1,
			'max_bin': 200,
			'od_wait': 65,
			'one_hot_max_size': 50,
			'grow_policy': 'Depthwise',
			'bootstrap_type': 'Bernoulli',
			'od_type': 'Iter',
			'eval_metric': 'AUC',
			'loss_function': 'Logloss',
			'task_type': self.device.upper(),
			'random_state': self.random_state,
		}
		cb_sym_params = cb_params.copy()
		cb_sym_params['grow_policy'] = 'SymmetricTree'
		cb_loss_params = cb_params.copy()
		cb_loss_params['grow_policy'] = 'Lossguide'

		cb_params2 = cb_params.copy()
		cb_params2['learning_rate'] = 0.01
		cb_params2['depth'] = 8
		cb_params3 = {
			'iterations': self.n_estimators,
			'random_strength': 0.5783342241486167,
			'one_hot_max_size': 10,
			'max_bin': 150,
			'learning_rate': 0.177,
			'l2_leaf_reg': 0.705662073971363,
			'grow_policy': 'SymmetricTree',
			'depth': 5,
			'max_bin': 200,
			'od_wait': 65,
			'bootstrap_type': 'Bayesian',
			'od_type': 'Iter',
			'eval_metric': 'AUC',
			'loss_function': 'Logloss',
			'task_type': self.device.upper(),
			'random_state': self.random_state,
		}
		cb_params4 = cb_params.copy()
		cb_params4['learning_rate'] = 0.01
		cb_params4['depth'] = 12
		dt_params = {'min_samples_split': 30, 'min_samples_leaf': 10, 'max_depth': 8, 'criterion': 'gini'}
		models = {
			'xgb': xgb.XGBClassifier(**xgb_params),
			#            'xgb2': xgb.XGBClassifier(**xgb_params2),
			#            'xgb3': xgb.XGBClassifier(**xgb_params3),
			#            'xgb4': xgb.XGBClassifier(**xgb_params4),
			#            'lgb': lgb.LGBMClassifier(**lgb_params),
			#             'lgb2': lgb.LGBMClassifier(**lgb_params2),
			#             'lgb3': lgb.LGBMClassifier(**lgb_params3),
			#             'lgb4': lgb.LGBMClassifier(**lgb_params4),
			'cat': CatBoostClassifier(**cb_params),
			#            'cat2': CatBoostClassifier(**cb_params2),
			#             'cat3': CatBoostClassifier(**cb_params3),
			#             'cat4': CatBoostClassifier(**cb_params4),
			"cat_sym": CatBoostClassifier(**cb_sym_params),
			#             "cat_loss": CatBoostClassifier(**cb_loss_params),
			#             'hist_gbm' : HistGradientBoostingClassifier (max_iter=300, learning_rate=0.001,  max_leaf_nodes=80,
			#            max_depth=6,random_state=self.random_state),#class_weight=class_weights_dict,
			#             'gbdt': GradientBoostingClassifier(max_depth=6,  n_estimators=1000,random_state=self.random_state),
			#             'lr': LogisticRegression(),
			#             'rf': RandomForestClassifier(max_depth= 9,max_features= 'auto',min_samples_split= 10,
			#                                                           min_samples_leaf= 4,  n_estimators=500,random_state=self.random_state),
			#            'svc': SVC(gamma="auto", probability=True),
			#             'knn': KNeighborsClassifier(n_neighbors=5),
			#             'mlp': MLPClassifier(random_state=self.random_state, max_iter=1000),
			#             'etr':ExtraTreesClassifier(min_samples_split=55, min_samples_leaf= 15, max_depth=10,
			#                                        n_estimators=200,random_state=self.random_state),
			#             'dt' :DecisionTreeClassifier(**dt_params,random_state=self.random_state),
			#             'ada': AdaBoostClassifier(random_state=self.random_state),

		}
		return models


class OptunaWeights:
	def __init__(self, random_state, n_trials=5000):
		self.study = None
		self.weights = None
		self.random_state = random_state
		self.n_trials = n_trials

	def _objective(self, trial, y_true, y_preds):
		weights = [trial.suggest_float(f'weight{n}', 0, 1) for n in range(len(y_preds))]
		weighted_pred = np.average(np.array(y_preds).T, axis=1, weights=weights)
		auc_score = roc_auc_score(y_true, weighted_pred)
		log_loss_score = log_loss(y_true, weighted_pred)
		return auc_score

	def fit(self, y_true, y_preds):
		optuna.logging.set_verbosity(optuna.logging.ERROR)
		sampler = optuna.samplers.CmaEsSampler(seed=self.random_state)
		pruner = optuna.pruners.HyperbandPruner()
		self.study = optuna.create_study(sampler=sampler, \
										 pruner=pruner, \
										 study_name='OptunaWeights', \
										 direction='maximize')
		objective_partial = partial(self._objective, \
									y_true=y_true, \
									y_preds=y_preds)
		self.study.optimize(objective_partial, n_trials=self.n_trials)
		self.weights = [self.study.best_params[f'weight{n}'] for n in range(len(y_preds))]

	def predict(self, y_preds):
		assert self.weights is not None, 'OptunaWeights error, must be fitted before .'
		weighted_pred = np.average(np.array(y_preds).T, axis=1, weights=self.weights)
		return weighted_pred

	def fit_predict(self, y_true, y_preds):
		self.fit(y_true, y_preds)
		return self.predict(y_preds)

	def weights(self):
		return self.weights


import gc


def fit_model(X_train, X_test, y_train):
	kfold = True
	n_splits = 1 if not kfold else 5
	random_state = 2023
	random_state_list = [42]
	n_estimators = 9999
	early_stopping_rounds = 300
	verbose = False
	splitter = Splitter(kfold, kfold, n_splits=n_splits)
	test_predss = np.zeros(X_test.shape[0])
	y_train_pred = y_train.copy()
	ensemble_score = []
	weights = []
	trained_models = {'xgb': [], 'lgb': []}
	for i, (X_train_, X_val, y_train_, y_val) in enumerate(
			splitter.split_data(X_train, y_train, random_state_list=random_state_list)):
		n = i % n_splits
		m = i // n_splits
		classifier = Classifier(n_estimators, device, random_state)
		models = classifier.models
		oof_preds = []
		test_preds = []
		for name, model in models.items():
			if ('cat' in name) or ("lgb" in name) or ("xgb" in name):
				if 'lgb' in name:  # categorical_feature=cat_features
					model.fit(X_train_, y_train_, eval_set=[(X_val, y_val)])
				elif 'cat' in name:
					model.fit(X_train_, y_train_, eval_set=[(X_val, y_val)], \
							  early_stopping_rounds=early_stopping_rounds, \
							  verbose=verbose)
				else:
					model.fit(X_train_, y_train_)
				test_pred = model.predict_proba(X_test)[:, 1]
				y_val_pred = models.predict_proba(X_val)[:, 1]
				score = roc_auc_score(y_val, y_val_pred.reshape(-1, 1))
				#         score = accuracy_score(y_val, acc_cutoff_class(y_val, y_val_pred))
				print(f'{name} [FOLD-{n} SEED-{random_state_list[m]}] ROC AUC score: {score:.5f}')
				oof_preds.append(y_val_pred)
				test_preds.append(test_pred)
				if name in trained_models.keys():
					trained_models[f'{name}'].append(deepcopy(models))
			optweights = OptunaWeights(random_state=random_state)
			y_val_pred = optweights.fit_predict(y_val.values, oof_preds)
			score = roc_auc_score(y_val, y_val_pred.reshape(-1, 1))
			print(f'Ensemble [FOLD -{n} seed - {random_state_list[m]}------------->ROC AUC score {score:.5f}')
			ensemble_score.append(score)
			weights.append(optweights.weights)
			test_preds += optweights.predict(test_preds) / (n_splits * len(random_state_list))
			y_train_pred.loc[y_val.index] = np.array(y_val_pred)
			gc.collect()
		# Calculate the mean ROC AUC  score of the ensemble
		mean_score = np.mean(ensemble_score)
		std_score = np.std(ensemble_score)
		print(f'Ensemble ROC AUC score {mean_score:.5f} + {std_score:.5f}')
		print('---- Model WEights')
		mean_weights = np.mean(weights, axis=0)
		std_weights = np.std(weights, axis=0)
		for name, mean_weight, std_weight in zip(models.keys(), mean_weights, std_weights):
			print(f'{name} : {mean_weight:.5f} + {std_weight:.5f}')
		print(f'Overall OFF Preds AUC SCORE {roc_auc_score(y_train, y_train_pred)}')
		print('-' * 100)
		return test_predss


def post_processor(train, test):
	cols = test.columns.tolist()
	train_cop = train_copy()
	test_cop = test.copy()
	drop_cols = []
	for i, feature in enumerate(cols):
		for j in range(i + 1, len(cols)):
			if sum(abs(train_cop[feature] - train_cop[cols[j]])) == 0:
				if cols[j] not in drop_cols:
					drop_cols.append(cols[j])
	print(drop_cols)
	train_cop.drop(columns=drop_cols, inplace=True)
	test_cop.drop(columns=drop_cols, inplace=True)
	return train_cop, test_cop


submission = pd.read_csv("../input/sample_submission.csv")
submission.head()

count = 0
for col in target:
	train_temp = train[test.columns.tolist() + [col]]
	test_temp = test.copy()
	train_temp, test_temp = cat_encoding(train_temp, test_temp, col)
	final_features = test.columns.tolist()
	sc = StandardScaler()
	train_scaled = train_temp.copy()
	test_scaled = test_temp.copy()

	train_scaled[final_features] = sc.fit_transform(train[final_features])
	test_scaled[final_features] = sc.transform(test[final_features])

	train_cop, test_cop = train_scaled, test_scaled
	X_train = train_cop.drop(columns=[col])
	y_train = train_cop[col]

	X_test = test_cop.copy()
	test_predss = fit_model(X_train, X_test, y_train)
	submission[col] = test_predss
	count += 1
	print(f'Column {col} ,loop # {count}')

submission.to_csv("submission_pure.csv", index=False)
submission.head()

sub1 = pd.read_csv("/kaggle/input/multiclass-feature-engineering-thoughts/submission.csv")
sub2 = pd.read_csv("/kaggle/input/ps4e03-multi-class-lightgbm/submission.csv")
sub_list = [sub1, sub2, submission]
weights = [1, 1, 1]
weighted_list = [item for sublist, weight in zip(sub_list, weights) for item in [sublist] * weight]


def ensemble_mean(sub_list, cols, mean="AM"):
	sub_out = sub_list[0].copy()
	if mean == "AM":
		for col in cols:
			sub_out[col] = sum(df[col] for df in sub_list) / len(sub_list)
	elif mean == "GM":
		for df in sub_list[1:]:
			for col in cols:
				sub_out[col] *= df[col]
		for col in cols:
			sub_out[col] = (sub_out[col]) ** (1 / len(sub_list))
	elif mean == 'HM':
		for col in cols:
			sub_out[col] = len(sub_list) / sum(1 / df[col] for df in sub_list)
	sub_out[cols] = sub_out[cols].div(sub_out[cols].sum(axis=1), axis=0)
	return sub_out


sub_ensemble = ensemble_mean(weighted_list, target, mean='AM')
sub_ensemble.to_csv('submission.csv', index=False)
print(sub_ensemble.head())
