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
import random
from random import randint, uniform
import gc
from sklearn.feature_selection import f_classif
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler, \
	PowerTransformer, FunctionTransformer
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from itertools import combinations
from sklearn.impute import SimpleImputer
import xgboost as xg
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_log_error, \
	mean_squared_error, \
	roc_auc_score, \
	accuracy_score, \
	f1_score, \
	precision_recall_curve, log_loss
from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer
from gap_statistic.optimalK import OptimalK
from scipy import stats
import statsmodels.api as sm
from sklearn.base import BaseEstimator, TransformerMixin
import seaborn as sns

sns.set(style='darkgrid', font_scale=1.4)
import optuna
import xgboost as xgb
import lightgbm as lgb
from category_encoders import OneHotEncoder, OrdinalEncoder, CountEncoder, CatBoostEncoder
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier, \
	GradientBoostingClassifier, ExtraTreesClassifier, AdaBoostClassifier
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.naive_bayes import GaussianNB
from sklearn.experimental import enable_hist_gradient_boosting
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
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD
# Suppress warnings
import warnings

warnings.filterwarnings('ignore')
import pandas as pd

pd.pandas.set_option('display.max_columns', None)

train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
original = pd.read_csv("../input/SteelPlatesFaults.csv")

train.drop(columns=["id"], inplace=True)
test.drop(columns=["id"], inplace=True)

train_copy = train.copy()
test_copy = test.copy()
original_copy = original.copy()

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


# fig, axes = plt.subplots(num_rows, num_cols, figsize=(21, 5 * num_rows))  # Adjust the figure size as needed

# for i, feature in enumerate(cont_cols):
# 	row = i // num_cols
# 	col = i % num_cols
# 	ax = axes[row, col] if num_rows > 1 else axes[col]
# 	sns.histplot(train_copy[feature], \
# 				 kde=True, color=colors[0], \
# 				 label='Train', alpha=0.5, bins=30, ax=ax)
# 	sns.histplot(test_copy[feature], \
# 				 kde=True, color=colors[1], \
# 				 label='Test', alpha=0.5, bins=30, ax=ax)
# 	sns.histplot(original[feature], \
# 				 kde=True, color=colors[2], \
# 				 label='Original', alpha=0.5, bins=30, ax=ax)
# 	ax.set_title(f'Distribution of {feature}')
# 	ax.set_xlabel(feature)
# 	ax.set_ylabel('Frequency')
# 	ax.legend()


# if num_plots % num_cols != 0:
#     for j in range(num_plots % num_cols, num_cols):
#         axes[-1, j].axis('off')

# plt.tight_layout()
# plt.savefig('1.png')
def OHE(train_df, test_df, cols, target):
	'''
	   Function for one hot encoding, it first combined the data so that no category
	   is missed and
	   the category with least frequency can be dropped because of redunancy
	   '''
	combined = pd.concat([train_df, test_df], axis=0)
	for col in cols:
		one_hot = pd.get_dummies(combined[col]).astype(int)
		counts = combined[col].value_counts()
		min_count_category = counts.idxmin()
		one_hot = one_hot.drop(min_count_category, axis=1)
		one_hot.columns = [str(f) + col for f in one_hot.columns]
		combined = pd.concat([combined, one_hot], axis="columns")
		combined = combined.loc[:, ~combined.columns.duplicated()]


cat_cols = [f for f in test.columns if test[f].nunique() / test.shape[0] * 100 < 5 and test[f].nunique() > 2]
print(test[cat_cols].nunique())


def nearest_val(target):
	return min(common, key=lambda x: abs(x - target))


global cat_cols_updated
cat_cols_updated = []
for col in cat_cols:
	train[f"{col}_cat"] = train[col]
	test[f"{col}_cat"] = test[col]
	cat_cols_updated.append(f"{col}_cat")
	uncommon = list(
		(set(test[col].unique()) | set(train[col].unique())) - (set(test[col].unique()) & set(train[col].unique())))
	if uncommon:
		common = list(set(test[col].unique()) & set(train[col].unique()))
		train[f"{col}_cat"] = train[col].apply(nearest_val)
		test[f"{col}_cat"] = test[col].apply(nearest_val)


def high_freq_ohe(train, test, extra_cols, target, n_limit=50):
	pass


def cat_encoding(train, test, target):
	global overall_best_score
	global overall_best_score
	table = PrettyTable()
	table.field_names = ['Feature', 'Encoded Features', 'Log Loss Score']
	train_copy = train.copy()
	test_copy = test.copy()
	train_dum = train.copy()
	for feature in cat_cols_updated:
		# print(feature)
		# cat_labels = train_dum.groupby([feature])[target].mean().sort_values().index
		# cat_labels2 = {k: i for i, k in enumerate(cat_labels, 0)}
		# train_copy[feature + '_target'] = train[feature].map(cat_labels2)
		# test_copy[feature + '_target'] = test[feature].map(cat_labels2)
		dic = train[feature].value_counts().to_dict()
		train_copy[feature + '_count'] = train[feature].map(dict)
		test_copy[feature + '_count'] = test[feature].map(dic)
		dic2 = train[feature].value_counts().to_dict()
		list1 = np.arange(len(dic2.values()))
		dic3 = dict(zip(list(dic2.keys()), list1))
		train_copy[feature + '_count_label'] = train[feature].replace(dic3).astype(int)
		test_copy[feature + '_count_label'] = test[feature].replace(dic3).astype(int)
		temp_cols = [feature + '_count', feature + '_count_label']
		if train_copy[feature].nunique() <= 5:
			train_copy[feature] = train_copy[feature].astype(str) + "_" + feature
			test_copy[feature] = test_copy[feature].astype(str) + "_" + feature
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
													   random_state=42)
				model.fit(X_train, y_train)
				y_pred = model.predict_proba(x_val)[:, 1]
				auc.append(roc_auc_score(y_val, y_pred))
			auc_scores.append((f, np.mean(auc)))
		best_col, best_auc = sorted(auc_scores, key=lambda x: x[1], reverse=True)[0]
		corr = train_copy[temp_cols].corr(method='pearson')
		corr_with_best_col = corr[best_col]
		cols_to_drop = [f for f in temp_cols if corr_with_best_col[f] > 0.5 and f != best_col]
		final_selection = [f for f in temp_cols if f not in cols_to_drop]
		if cols_to_drop:
			train_copy = train_copy.drop(columns=cols_to_drop)
			test_copy = test_copy.drop(columns=cols_to_drop)
		table.add_row([feature, best_col, best_auc])
	#         print(feature)
	#     print(table)
	return train_copy, test_copy


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
