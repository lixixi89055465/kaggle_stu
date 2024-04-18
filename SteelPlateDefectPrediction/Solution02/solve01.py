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


cat_cols = [f for f in test.columns if test[f].nunique() / test.shape[0] * 100 < 5 and test[f].nunique() > 2]
test[cat_cols].nunique()


def nearst_val(target):
	return min(common, key=lambda x: abs(x - target))


global cat_cols_updated
cat_cols_updated = []
for col in cat_cols:
	train[f'{col}_cat'] = train[col]
	test[f'{col}_cat'] = test[col]
	cat_cols_updated.append(f'{col}_cat')
	uncommon = list(
		(set(test[col].unique()) | set(train[col].unique())) - (set(test[col].unique()) & set(train[col].unique())))
	if uncommon:
		common = list(set(test[col].unique()) & set(train[col].unique()))
		train[f'{col}_cat'] = train[col].apply(nearst_val)
		test[f'{col}_cat'] = test[col].apply(nearst_val)


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
		ext_keys = [f[0] for f in orderd.items() if f[1] < 50]
		rare_key_map = dict(zip(rare_keys, np.full(len(rare_keys), 9999)))
		train_copy[col]
		train_copy[col].replace(rare_key_map)
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
		train_copy=train_copy.drop(columns=[feature])

