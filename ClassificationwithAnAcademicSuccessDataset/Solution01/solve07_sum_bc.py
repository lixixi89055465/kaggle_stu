'''
Hello and Welcome to Kaggle, the online Data Science Community to learn, share, and compete. Most beginners get lost in the field, because they fall into the black box approach, using libraries and algorithms they don't understand. This tutorial will give you a 1-2-year head start over your peers, by providing a framework that teaches you how-to think like a input scientist vs what to think/code. Not only will you be able to submit your first competition, but you’ll be able to solve any problem thrown your way. I provide clear explanations, clean code, and plenty of links to resources. Please Note: This Kernel is still being improved. So check the Change Logs below for updates. Also, please be sure to upvote, fork, and comment and I'll continue to develop. Thanks, and may you have "statistically significant" luck!
'''
import sys
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import seaborn as sns
from pandas.plotting import scatter_matrix

# Configure Visualization Defaults
# %matplotlib inline = show plots in Jupyter Notebook browser
# %matplotlib inline
mpl.style.use('ggplot')
sns.set_style('white')
pylab.rcParams['figure.figsize'] = 12, 8
print('Python version:{}'.format(sys.version))
import pandas as pd

print("Pandas version:{}".format(pd.__version__))
import matplotlib

print('matplotlib.version {}'.format(matplotlib.__version__))

import numpy as np

print("Numpy version {}".format(np.__version__))

import scipy as sp

print("scipy version: {}".format(sp.__version__))

import IPython
from IPython import display

print("Ipython version :{} ".format(IPython.__version__))

import sklearn

print('Sklear version: {}'.format(sklearn.__version__))
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

import random
import time

import warnings

warnings.filterwarnings('ignore')
print('-' * 25)
from subprocess import check_output

# print(check_output(['ls', '../input/']).decode('utf8'))

import pandas as pd

data_raw = pd.read_csv('../input/playground-series-s4e6/train.csv')
data_val = pd.read_csv('../input/playground-series-s4e6/test.csv')
# #to play with our input we'll create a copy
# #remember python assignment or equal passes by reference vs values, so we use the copy function: https://stackoverflow.com/questions/46327494/python-pandas-dataframe-copydeep-false-vs-copydeep-true-vs
# data1 = data_raw.copy(deep=True)
# data_cleaner = [data1, data_val]
print("2" * 100)
print(data_raw.info())
print(data_raw.head())
print(data_raw.sample(9))
# Duplicates
import numpy as np

print("3" * 100)
print(
	f'Duplicates in train set :{data_raw.duplicated().sum()},'
	f'{np.round(100 * data_raw.duplicated().sum() / len(data_raw), 1)})')
print(
	f'Duplicates in test set :{data_raw.duplicated().sum()},'
	f'{np.round(100 * data_val.duplicated().sum() / len(data_raw), 1)})')

# print('Train columns with null value:\n', data1.isnull().sum())
# print("-" * 100)
# print('Test/Validation columns with null value :\n', data_val.isnull().sum())
# print("- " * 100)
print(data_raw.nunique())
print(data_raw.dtypes)
# Expenditure features
exp_feats = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
# Categorical feature
cat_feats = ['HomePlanet', 'CryoSleep', 'Destination', 'VIP']
# Qualitative features
qual_feats = ['PassengerId', 'Cabin', 'Name']
#
# for dataset in data_cleaner:
# 	dataset['Age_group'] = np.nan
# 	dataset.loc[dataset['Age'] <= 12, 'Age_group'] = 'Age_0-12'
# 	dataset.loc[(dataset['Age'] > 12) & (dataset['Age'] < 18), 'Age_group'] = 'Age_13-17'
# 	dataset.loc[(dataset['Age'] >= 18) & (dataset['Age'] <= 25), 'Age_group'] = 'Age_18-25'
# 	dataset.loc[(dataset['Age'] > 25) & (dataset['Age'] <= 30), 'Age_group'] = 'Age_26-30'
# 	dataset.loc[(dataset['Age'] > 30) & (dataset['Age'] <= 50), 'Age_group'] = 'Age_31-50'
# 	dataset.loc[dataset['Age'] > 50, 'Age_group'] = 'Age_51+'


# for dataset in data_cleaner:
# 	dataset['Group'] = dataset['PassengerId'].apply(lambda x: x.split('_')[0]).astype(int)
# 	dvc = dataset['Group'].value_counts()
# 	dataset['Group_size'] = dataset['Group'].map(lambda x: dvc[x])
# dataset['Solo'] = (dataset['Group_size'] == 1).astype(int)

## New features
# data_raw['Solo'] = (data_raw['Group_size'] == 1).astype(int)
# data_val['Solo'] = (data_val['Group_size'] == 1).astype(int)

# print('1' * 100)
# for dataset in data_cleaner:
# 	dataset['Solo'] = (dataset['Group_size'] == 1).astype(int)
#
## TODO 离散分批特征
# for dataset in data_cleaner:
# 	# New feature - training set
# 	dataset['Cabin_region1'] = (dataset['Cabin_number'] < 300).astype(int)
# 	dataset['Cabin_region2'] = ((dataset['Cabin_number'] >= 300) & (dataset['Cabin_number'] < 600)).astype(int)
# 	dataset['Cabin_region3'] = ((dataset['Cabin_number'] >= 600) & (dataset['Cabin_number'] < 900)).astype(int)
# 	dataset['Cabin_region4'] = ((dataset['Cabin_number'] >= 900) & (dataset['Cabin_number'] < 1200)).astype(int)
# 	dataset['Cabin_region5'] = ((dataset['Cabin_number'] >= 1200) & (dataset['Cabin_number'] < 1500)).astype(int)
# 	dataset['Cabin_region6'] = ((dataset['Cabin_number'] >= 1500) & (dataset['Cabin_number'] < 1500)).astype(int)
# 	dataset['Cabin_region7'] = (dataset['Cabin_number'] >= 1800).astype(int)

# plt.figure(figsize=(10, 4))
# TODO 求和特征
# data1['Cabin_region_plot'] = (
# 		data1['Cabin_region1'] + 2 * data1['Cabin_region2'] + 3 * data1['Cabin_region3'] + 4 * data1[
# 	'Cabin_region4'] + 5 * data1['Cabin_region5'] + 6 * data1['Cabin_region6'] + 7 * data1['Cabin_region7']).astype(int)
# sns.countplot(input=data1, x='Cabin_region_plot', hue='Transported')
# plt.title('Cabin region ')
# plt.show()
# data1.drop('Cabin_region_plot', axis=1, inplace=True)

print('1' * 100)  # TODO
##TODO 创造 新特征
# for data in data_cleaner:
# 	# Joint distribution of Surname and Cabin side
# 	SCS_gb = data[data['Group_size'] > 1].groupby(['Surname', 'Cabin_side'])['Cabin_side'].size().unstack().fillna(0)
# 	# Ratio of sides
# 	# TODO 特征比例
# 	SCS_gb['Ratio'] = SCS_gb['P'] / (SCS_gb['P'] + SCS_gb['S'])


# TODO One-hot encode cabin regions
# data['Cabin_region1'] = (data['Cabin_number'] < 300).astype(int)
# data['Cabin_region2'] = ((data['Cabin_number'] >= 300) & (data['Cabin_number'] < 600)).astype(int)
# data['Cabin_region3'] = ((data['Cabin_number'] >= 600) & (data['Cabin_number'] < 900)).astype(int)
# data['Cabin_region4'] = ((data['Cabin_number'] >= 900) & (data['Cabin_number'] < 1200)).astype(int)
# data['Cabin_region5'] = ((data['Cabin_number'] >= 1200) & (data['Cabin_number'] < 1500)).astype(int)
# data['Cabin_region6'] = ((data['Cabin_number'] >= 1500) & (data['Cabin_number'] < 1800)).astype(int)
# data['Cabin_region7'] = (data['Cabin_number'] >= 1800).astype(int)

print('2' * 100)
##TODO  特征分段
# for data in data_cleaner:
# 	data.loc[data['Age'] <= 12, 'Age_group'] = 'Age_0-12'
# 	data.loc[(data['Age'] > 12) & (data['Age'] < 18), 'Age_group'] = 'Age_13-17'
# 	data.loc[(data['Age'] >= 18) & (data['Age'] <= 25), 'Age_group'] = 'Age_18-25'
# 	data.loc[(data['Age'] > 25) & (data['Age'] <= 30), 'Age_group'] = 'Age_26-30'
# 	data.loc[(data['Age'] > 30) & (data['Age'] <= 50), 'Age_group'] = 'Age_31-50'
# 	data.loc[data['Age'] > 50, 'Agegroup'] = 'Age_51+'

##TODO log 特征
# for data in data_cleaner:
# 	data['Expenditure'] = data[exp_feats].sum(axis=1)
# 	data['No_spending'] = (data['Expenditure'] == 0).astype(int)
# 	# input.isna().sum()
# 	for col in ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'Expenditure']:
# 		data[col] = np.log(1 + data[col])
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, RepeatedStratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from xgboost import XGBClassifier


##Feature importance
def objective(trial):
	max_depth = trial.suggest_int('max_depth', 3, 10)
	n_estimators = trial.suggest_int('n_estimators', 100, 2000)
	gamma = trial.suggest_float('gamma', 0, 1)
	reg_alpha = trial.suggest_float('reg_alpha', 0, 2)
	reg_lambda = trial.suggest_float('reg_lambda', 0, 2)
	min_child_weight = trial.suggest_int('min_child_weight', 0, 10)
	subsample = trial.suggest_float('subsample', 0, 1)
	colsample_bytree = trial.suggest_float('colsample_bytree', 0, 1)
	learning_rate = trial.suggest_float('learning_rate', 0.01, 1)

	print('Training the model with', train.shape[1], 'features')

	params = {'n_estimators': n_estimators,
			  'learning_rate': learning_rate,
			  'gamma': gamma,
			  'reg_alpha': reg_alpha,
			  'reg_lambda': reg_lambda,
			  'max_depth': max_depth,
			  'min_child_weight': min_child_weight,
			  'subsample': subsample,
			  'colsample_bytree': colsample_bytree,
			  'eval_metric': 'logloss'}  # Using logloss for binary classification

	clf = XGBClassifier(**params,
						booster='gbtree',
						objective='binary:logistic',  # Binary classification objective
						verbosity=0)

	cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
	cv_results = cross_val_score(clf, train, data_val, cv=cv, scoring='roc_auc')  # Using roc_auc scoring

	validation_score = np.mean(cv_results)
	return validation_score


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
	# df_v = df[col].value_counts()
	end_mem = df.memory_usage().sum() / 1024 ** 2
	print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
	print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
	return df


import re

train = data_raw.rename(columns=lambda x: re.sub('[^A-Za-z0-9_]+', '', x))
data_val = data_val.rename(columns=lambda x: re.sub('[^A-Za-z0-9_]+', '', x))

xgb_best_params_for_y1 = {'max_depth': 5, \
						  'n_estimators': 1627, \
						  'gamma': 0.8952807768735265,
						  'reg_alpha': 1.6314226873472901, \
						  'reg_lambda': 1.7229132141868826, \
						  'min_child_weight': 9,
						  'subsample': 0.9885054042421748, \
						  'colsample_bytree': 0.22439719563481197, \
						  'learning_rate': 0.10650804734533341}



data1_x_calc = ['Maritalstatus', 'Applicationmode', 'Applicationorder', 'Course',
				'Daytimeeveningattendance', 'Previousqualification',
				'Previousqualificationgrade', 'Nacionality', 'Mothersqualification',
				'Fathersqualification', 'Mothersoccupation', 'Fathersoccupation',
				'Admissiongrade', 'Displaced', 'Educationalspecialneeds', 'Debtor',
				'Tuitionfeesuptodate', 'Gender', 'Scholarshipholder', 'Ageatenrollment',
				'International', 'Curricularunits1stsemcredited',
				'Curricularunits1stsemenrolled', 'Curricularunits1stsemevaluations',
				'Curricularunits1stsemapproved', 'Curricularunits1stsemgrade',
				'Curricularunits1stsemwithoutevaluations',
				'Curricularunits2ndsemcredited', 'Curricularunits2ndsemenrolled',
				'Curricularunits2ndsemevaluations', 'Curricularunits2ndsemapproved',
				'Curricularunits2ndsemgrade', 'Curricularunits2ndsemwithoutevaluations',
				'Unemploymentrate', 'Inflationrate', 'GDP']
cur_sum = 'curr_sum'

cur_col = ['Curricularunits1stsemcredited',
		   'Curricularunits1stsemenrolled', 'Curricularunits1stsemevaluations',
		   'Curricularunits1stsemapproved', 'Curricularunits1stsemgrade',
		   'Curricularunits1stsemwithoutevaluations',
		   'Curricularunits2ndsemcredited', 'Curricularunits2ndsemenrolled',
		   'Curricularunits2ndsemevaluations', 'Curricularunits2ndsemapproved',
		   'Curricularunits2ndsemgrade', 'Curricularunits2ndsemwithoutevaluations',
		   ]
train[cur_sum] = train[cur_col].sum(axis=1)
data_val[cur_sum] = data_val[cur_col].sum(axis=1)
data1_x_calc += [cur_sum]

train = reduce_mem_usage(train)
data_val = reduce_mem_usage(data_val)

Target = 'Target'
labelEncoder = LabelEncoder()
train[Target] = labelEncoder.fit_transform(train[Target])
xgb_model_for_y1 = XGBClassifier(**xgb_best_params_for_y1)
result = xgb_model_for_y1.fit(train[data1_x_calc], train[Target])

from sklearn import model_selection

print(data_raw.describe(include='all'))
train_x, test1_x, train1_y, test1_y = model_selection.train_test_split(train[data1_x_calc], train[Target],
																	   random_state=0)

print('6' * 100)

# Machine Learning Algorithm (MLA) Selection and Initialization
from sklearn import ensemble, gaussian_process, \
	linear_model, naive_bayes, \
	neighbors, svm, tree, discriminant_analysis
from xgboost import XGBClassifier

# MLA = [
# 	# Ensemble Methods
# 	ensemble.AdaBoostClassifier(),
# 	ensemble.BaggingClassifier(),
# 	ensemble.ExtraTreesClassifier(),
# 	# # TODO
# 	ensemble.GradientBoostingClassifier(),
# 	ensemble.RandomForestClassifier(),
# 	#
# 	# # Gaussian Processes
# 	# gaussian_process.GaussianProcessClassifier(),
# 	# # GLM
# 	linear_model.LogisticRegressionCV(),
# 	linear_model.PassiveAggressiveClassifier(),
# 	linear_model.RidgeClassifierCV(),
# 	linear_model.SGDClassifier(),
# 	linear_model.Perceptron(),
#
# 	# Navies Bayer
# 	naive_bayes.BernoulliNB(),
# 	naive_bayes.GaussianNB(),
# 	# Nearest Neighbor
# 	neighbors.KNeighborsClassifier(),
# 	#
# 	# SVM
# 	svm.SVC(probability=True),
# 	svm.NuSVC(probability=True),
# 	svm.LinearSVC(),
# 	# Trees
# 	tree.DecisionTreeClassifier(),
# 	tree.ExtraTreeClassifier(),
# 	# Discriminat Analysisi
# 	discriminant_analysis.LinearDiscriminantAnalysis(),
# 	discriminant_analysis.QuadraticDiscriminantAnalysis(),
# 	#xgboost: http://xgboost.readthedocs.io/en/latest/model.html
# 	# TODO
# 	XGBClassifier()
# ]
# split dataset in cross-validation with this splitter class:
# http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.ShuffleSplit.html#sklearn.model_selection.ShuffleSplit
# note: this is an alternative to train_test_split
cv_split = model_selection.ShuffleSplit(n_splits=10, test_size=.3, train_size=.6, random_state=0)
# create table to compare MLA metrics
MLA_columns = ['MLA Name', 'MLA Parameters', 'MLA Train Accuracy Mean', \
			   'MLA Test Accuracy Mean', 'MLA Test Accuracy 3*STD', 'MLA Time']
MLA_compare = pd.DataFrame(columns=MLA_columns)

# create table to compare MLA predictions
MLA_predict = train[Target]

# index through MLA and save performance to table
# row_index = 0
# for alg in MLA:
# 	MLA_name = alg.__class__.__name__
# 	MLA_compare.loc[row_index, 'MLA Name'] = MLA_name
# 	MLA_compare.loc[row_index, 'MLA Parameters'] = str(alg.get_params())
# 	# score model with cross validation: http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_validate.html#sklearn.model_selection.cross_validate
# 	cv_results = model_selection.cross_validate(
# 		alg,  #
# 		train[data1_x_calc],  #
# 		train[Target],  #
# 		cv=cv_split,  #
# 		return_train_score=True
# 	)
# 	MLA_compare.loc[row_index, 'MLA Time'] = cv_results['fit_time'].mean()
# 	MLA_compare.loc[row_index, 'MLA Train Accuracy Mean'] = cv_results['train_score'].mean()
# 	MLA_compare.loc[row_index, 'MLA Test Accuracy Mean'] = cv_results['test_score'].mean()
# 	# if this is a non-bias random sample, then +/-3 standard deviations (std) from the mean, should statistically capture 99.7% of the subsets
# 	MLA_compare.loc[row_index, 'MLA Test accuracy 3*STD'] = cv_results['test_score'].std() * 3
#
# 	# Save MLA predicttune_modelions - see section 6 for usage
# 	alg.fit(train[data1_x_calc], train[Target])
# 	MLA_predict[MLA_name] = alg.predict(train[data1_x_calc])
# 	row_index += 1

# print and sort table: https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.sort_values.html
MLA_compare.sort_values(by=['MLA Test Accuracy Mean'], ascending=False, inplace=True)

# model input
# Tree_Predict = mytree(data1)

# TODO
# print('Decision Tree Model Accuracy /Precision Score:{:.2f}%\n'.
# 	  format(metrics.accuracy_score(data1['Transported'], Tree_Predict) * 100))
# Accuracy Summary Report with http://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html#sklearn.metrics.classification_report
# Where recall score = (true positives)/(true positive + false negative) w/1 being best:http://scikit-learn.org/stable/modules/generated/sklearn.metrics.recall_score.html#sklearn.metrics.recall_score
# And F1 score = weighted average of precision and recall w/1 being best: http://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html#sklearn.metrics.f1_score
# print(metrics.classification_report(data1['Transported'], Tree_Predict))
# Plot Accuracy Summary
# Credit: http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
import itertools

# tune hyper-parameters: http://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html#sklearn.tree.DecisionTreeClassifier
param_grid = {'criterion': ['gini', 'entropy'],
			  # scoring methodology; two supported formulas for calculating information gain - default is gini
			  # 'splitter': ['best', 'random'], #splitting methodology; two supported strategies - default is best
			  'max_depth': [2, 4, 6, 8, 10, None],  # max depth tree can grow; default is none
			  # 'min_samples_split': [2,5,10,.03,.05], #minimum subset size BEFORE new split (fraction is % of total); default is 2
			  # 'min_samples_leaf': [1,5,10,.03,.05], #minimum subset size AFTER new split split (fraction is % of total); default is 1
			  # 'max_features': [None, 'auto'], #max features to consider when performing split; default none or all
			  'random_state': [0]
			  # seed or control random number generator: https://www.quora.com/What-is-seed-in-random-number-generation
			  }

# print(list(model_selection.ParameterGrid(param_grid)))

# choose best model with grid_search: #http://scikit-learn.org/stable/modules/grid_search.html#grid-search
# http://scikit-learn.org/stable/auto_examples/model_selection/plot_grid_search_digits.html
# tune_model = model_selection.GridSearchCV(tree.DecisionTreeClassifier(), param_grid=param_grid, scoring='roc_auc',
# 										  cv=cv_split, return_train_score=True)
# tune_model.fit(train[data1_x_calc], train[Target])
#
# # print(tune_model.cv_results_.keys())
# # print(tune_model.cv_results_['params'])
# print('AFTER DT Parameters: ', tune_model.best_params_)
# # print(tune_model.cv_results_['mean_train_score'])
# print("AFTER DT Training w/bin score mean: {:.2f}".format(
# 	tune_model.cv_results_['mean_train_score'][tune_model.best_index_] * 100))
# # print(tune_model.cv_results_['mean_test_score'])
# print("AFTER DT Test w/bin score mean: {:.2f}".format(
# 	tune_model.cv_results_['mean_test_score'][tune_model.best_index_] * 100))
# print("AFTER DT Test w/bin score 3*std: +/- {:.2f}".format(
# 	tune_model.cv_results_['std_test_score'][tune_model.best_index_] * 100 * 3))
# print('-' * 10)

# duplicates gridsearchcv
# tune_results = model_selection.cross_validate(tune_model, data1[data1_x_bin], data1[Target], cv  = cv_split)

# print('AFTER DT Parameters: ', tune_model.best_params_)
# print("AFTER DT Training w/bin set score mean: {:.2f}". format(tune_results['train_score'].mean()*100))
# print("AFTER DT Test w/bin set score mean: {:.2f}". format(tune_results['test_score'].mean()*100))
# print("AFTER DT Test w/bin set score min: {:.2f}". format(tune_results['test_score'].min()*100))
# print('-'*10)


# base model
# print('BEFORE DT RFE Training Shape Old: ', data1[data1_x_bin].shape)
# print('BEFORE DT RFE Training Columns Old: ', data1[data1_x_bin].columns.values)
#
# print("BEFORE DT RFE Training w/bin score mean: {:.2f}".format(base_results['train_score'].mean() * 100))
# print("BEFORE DT RFE Test w/bin score mean: {:.2f}".format(base_results['test_score'].mean() * 100))
# print("BEFORE DT RFE Test w/bin score 3*std: +/- {:.2f}".format(base_results['test_score'].std() * 100 * 3))
print('-' * 10)

# TODO
from sklearn import feature_selection

# TODO
# feature selection
# dtree_rfe = feature_selection.RFECV(dtree, step=1, scoring='accuracy', cv=cv_split)
# dtree_rfe.fit(data1[data1_x_bin], data1[Target])

# transform x&y to reduced features and fit new model
# alternative: can use pipeline to reduce fit and transform steps:
# http://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html
# X_rfe = data1[data1_x_bin].columns.values[dtree_rfe.get_support()]
# rfe_results = model_selection.cross_validate(dtree, \
# 											 data1[X_rfe], \
# 											 data1[Target], \
# 											 cv=cv_split, \
# 											 return_train_score=True)
# print(dtree_rfe.grid_scores_)
# print('AFTER DT RFE Training Shape New: ', data1[X_rfe].shape)
# print('AFTER DT RFE Training Columns New: ', X_rfe)
#
# print("AFTER DT RFE Training w/bin score mean: {:.2f}".format(rfe_results['train_score'].mean() * 100))
# print("AFTER DT RFE Test w/bin score mean: {:.2f}".format(rfe_results['test_score'].mean() * 100))
# print("AFTER DT RFE Test w/bin score 3*std: +/- {:.2f}".format(rfe_results['test_score'].std() * 100 * 3))
print('-' * 10)

# tune rfe model
# rfe_tune_model = model_selection.GridSearchCV(tree.DecisionTreeClassifier(), \
# 											  param_grid=param_grid, \
# 											  scoring='roc_auc', \
# 											  cv=cv_split, \
# 											  return_train_score=True)
# rfe_tune_model.fit(train[X_rfe], train[Target])

# print(rfe_tune_model.cv_results_.keys())
# print(rfe_tune_model.cv_results_['params'])
# print('AFTER DT RFE Tuned Parameters: ', rfe_tune_model.best_params_)
# # print(rfe_tune_model.cv_results_['mean_train_score'])
# print("AFTER DT RFE Tuned Training w/bin score mean: {:.2f}".format(
# 	rfe_tune_model.cv_results_['mean_train_score'][tune_model.best_index_] * 100))
# # print(rfe_tune_model.cv_results_['mean_test_score'])
# print("AFTER DT RFE Tuned Test w/bin score mean: {:.2f}".format(
# 	rfe_tune_model.cv_results_['mean_test_score'][tune_model.best_index_] * 100))
# print("AFTER DT RFE Tuned Test w/bin score 3*std: +/- {:.2f}".format(
# 	rfe_tune_model.cv_results_['std_test_score'][tune_model.best_index_] * 100 * 3))
# print('-' * 10)

# Graph MLA version of Decision Tree: http://scikit-learn.org/stable/modules/generated/sklearn.tree.export_graphviz.html
# import graphviz

# dot_data = tree.export_graphviz(dtree, out_file=None,
# 								feature_names=data1_x_bin, class_names=True,
# 								filled=True, rounded=True)
# graph = graphviz.Source(dot_data)
# graph

# compare algorithm predictions with each other, where 1 = exactly similar and 0 = exactly opposite
# there are some 1's, but enough blues and light reds to create a "super algorithm" by combining them
# correlation_heatmap(MLA_predict)
# why choose one model, when you can pick them all with voting classifier
# http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.VotingClassifier.html
# removed models w/o attribute 'predict_proba' required for
# vote classifier and models with a 1.0 correlation to
# another model
from sklearn.svm import SVC

vote_est = [
	# Ensemble Methods: http://scikit-learn.org/stable/modules/ensemble.html
	# ('ada', ensemble.AdaBoostClassifier()),# Score: 0.73907
	('bc', ensemble.BaggingClassifier(n_jobs=-1)),
	# ('etc', ensemble.ExtraTreesClassifier(n_jobs=-1)),
	# ('gbc', ensemble.GradientBoostingClassifier()),
	# ('rfc', ensemble.RandomForestClassifier()),
	# # Gaussian Processes: http://scikit-learn.org/stable/modules/gaussian_process.html#gaussian-process-classification-gpc
	# ('gpc', gaussian_process.GaussianProcessClassifier()),
	# # GLM: http://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
	# ('lr', linear_model.LogisticRegressionCV()),
	# # Navies Bayes: http://scikit-learn.org/stable/modules/naive_bayes.html
	# ('bnb', naive_bayes.BernoulliNB()),
	# ('gnb', naive_bayes.GaussianNB()),
	# # Nearest Neighbor: http://scikit-learn.org/stable/modules/neighbors.html
	# ('knn', neighbors.KNeighborsClassifier()),
	# # SVM: http://scikit-learn.org/stable/modules/svm.html
	# # ('svc', SVC(probability==True)),
	# ('svc', svm.SVC(probability=True)),
	# # xgboost: http://xgboost.readthedocs.io/en/latest/model.html
	# ('xgb', XGBClassifier())
]
# Hard Vote or majority rules
vote_hard = ensemble.VotingClassifier(estimators=vote_est, voting='hard')
vote_hard_cv = model_selection.cross_validate(vote_hard, train[data1_x_calc], train[Target], \
											  cv=cv_split, \
											  return_train_score=True)
vote_hard.fit(train[data1_x_calc], train[Target])
print("Hard Voting Training w/bin score mean: {:.2f}".format(vote_hard_cv['train_score'].mean() * 100))
print("Hard Voting Test w/bin score mean: {:.2f}".format(vote_hard_cv['test_score'].mean() * 100))
print("Hard Voting Test w/bin score 3*std: +/- {:.2f}".format(vote_hard_cv['test_score'].std() * 100 * 3))
print('-' * 10)

# Soft Vote or weighted probabilities

vote_soft = ensemble.VotingClassifier(estimators=vote_est, voting='soft')
vote_soft_cv = model_selection.cross_validate(vote_soft, \
											  train[data1_x_calc], \
											  train[Target], \
											  cv=cv_split, return_train_score=True)
vote_soft.fit(train[data1_x_calc], train[Target])

# print("Soft Voting Training w/bin score mean: {:.2f}".format(vote_soft_cv['train_score'].mean() * 100))
# print("Soft Voting Test w/bin score mean: {:.2f}".format(vote_soft_cv['test_score'].mean() * 100))
# print("Soft Voting Test w/bin score 3*std: +/- {:.2f}".format(vote_soft_cv['test_score'].std() * 100 * 3))
print('-' * 10)
# IMPORTANT: THIS SECTION IS UNDER CONSTRUCTION!!!!
# UPDATE: This section was scrapped for the next section; as it's more computational friendly.

# WARNING: Running is very computational intensive and time expensive code is written for experimental/developmental purposes and not production ready


# tune each estimator before creating a super model
# http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html
grid_n_estimator = [50, 100, 300]
grid_ratio = [.1, .25, .5, .75, 1.0]
grid_learn = [.01, .03, .05, .1, .25]
grid_max_depth = [2, 4, 6, None]
grid_min_samples = [5, 10, .03, .05, .10]
grid_criterion = ['gini', 'entropy']
grid_bool = [True, False]
grid_seed = [0]

vote_param = [{
	# #            #http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html
	# 'ada__n_estimators': grid_n_estimator,
	# 'ada__learning_rate': grid_ratio,
	# 'ada__algorithm': ['SAMME', 'SAMME.R'],
	# 'ada__random_state': grid_seed,
	#
	# # http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.BaggingClassifier.html#sklearn.ensemble.BaggingClassifier
	'bc__n_estimators': grid_n_estimator,
	'bc__max_samples': grid_ratio,
	'bc__oob_score': grid_bool,
	'bc__random_state': grid_seed,
	#
	# # http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesClassifier.html#sklearn.ensemble.ExtraTreesClassifier
	# 'etc__n_estimators': grid_n_estimator,
	# 'etc__criterion': grid_criterion,
	# 'etc__max_depth': grid_max_depth,
	# 'etc__random_state': grid_seed,
	#
	# # http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html#sklearn.ensemble.GradientBoostingClassifier
	# 'gbc__loss': ['deviance', 'exponential'],
	# 'gbc__learning_rate': grid_ratio,
	# 'gbc__n_estimators': grid_n_estimator,
	# 'gbc__criterion': ['friedman_mse', 'mse', 'mae'],
	# 'gbc__max_depth': grid_max_depth,
	# 'gbc__min_samples_split': grid_min_samples,
	# 'gbc__min_samples_leaf': grid_min_samples,
	# 'gbc__random_state': grid_seed,
	#
	# # http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html#sklearn.ensemble.RandomForestClassifier
	# 'rfc__n_estimators': grid_n_estimator,
	# 'rfc__criterion': grid_criterion,
	# 'rfc__max_depth': grid_max_depth,
	# 'rfc__min_samples_split': grid_min_samples,
	# 'rfc__min_samples_leaf': grid_min_samples,
	# 'rfc__bootstrap': grid_bool,
	# 'rfc__oob_score': grid_bool,
	# 'rfc__random_state': grid_seed,
	#
	# # http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegressionCV.html#sklearn.linear_model.LogisticRegressionCV
	# 'lr__fit_intercept': grid_bool,
	# 'lr__penalty': ['l1', 'l2'],
	# 'lr__solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
	# 'lr__random_state': grid_seed,
	#
	# # http://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.BernoulliNB.html#sklearn.naive_bayes.BernoulliNB
	# 'bnb__alpha': grid_ratio,
	# 'bnb__prior': grid_bool,
	# 'bnb__random_state': grid_seed,
	#
	# # http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html#sklearn.neighbors.KNeighborsClassifier
	# 'knn__n_neighbors': [1, 2, 3, 4, 5, 6, 7],
	# 'knn__weights': ['uniform', 'distance'],
	# 'knn__algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
	# 'knn__random_state': grid_seed,
	#
	# # http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC
	# # http://blog.hackerearth.com/simple-tutorial-svm-parameter-tuning-python-r
	# 'svc__kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
	# 'svc__C': grid_max_depth,
	# 'svc__gamma': grid_ratio,
	# 'svc__decision_function_shape': ['ovo', 'ovr'],
	# 'svc__probability': [True],
	# 'svc__random_state': grid_seed,
	#
	# # http://xgboost.readthedocs.io/en/latest/parameter.html
	# 'xgb__learning_rate': grid_ratio,
	# 'xgb__max_depth': [2, 4, 6, 8, 10],
	# 'xgb__tree_method': ['exact', 'approx', 'hist'],
	# 'xgb__objective': ['reg:linear', 'reg:logistic', 'binary:logistic'],
	# 'xgb__seed': grid_seed

}]

# Soft Vote with tuned models
# grid_soft = model_selection.GridSearchCV(estimator = vote_soft, param_grid = vote_param, cv = 2, scoring = 'roc_auc')
# grid_soft.fit(data1[data1_x_bin], data1[Target])

# print(grid_soft.cv_results_.keys())
# print(grid_soft.cv_results_['params'])
# print('Soft Vote Tuned Parameters: ', grid_soft.best_params_)
# print(grid_soft.cv_results_['mean_train_score'])
# print("Soft Vote Tuned Training w/bin set score mean: {:.2f}". format(grid_soft.cv_results_['mean_train_score'][tune_model.best_index_]*100))
# print(grid_soft.cv_results_['mean_test_score'])
# print("Soft Vote Tuned Test w/bin set score mean: {:.2f}". format(grid_soft.cv_results_['mean_test_score'][tune_model.best_index_]*100))
# print("Soft Vote Tuned Test w/bin score 3*std: +/- {:.2f}". format(grid_soft.cv_results_['std_test_score'][tune_model.best_index_]*100*3))
# print('-'*10)


# credit: https://rasbt.github.io/mlxtend/user_guide/classifier/EnsembleVoteClassifier/
# cv_keys = ('mean_test_score', 'std_test_score', 'params')
# for r, _ in enumerate(grid_soft.cv_results_['mean_test_score']):
#    print("%0.3f +/- %0.2f %r"
#          % (grid_soft.cv_results_[cv_keys[0]][r],
#             grid_soft.cv_results_[cv_keys[1]][r] / 2.0,
#             grid_soft.cv_results_[cv_keys[2]][r]))

# print('-'*10)

# WARNING: Running is very computational intensive and time expensive.
# Code is written for experimental/developmental purposes and not production ready!

# Hyperparameter Tune with GridSearchCV: http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html
grid_n_estimator = [10, 50, 100, 300]
grid_ratio = [.1, .25, .5, .75, 1.0]
grid_learn = [.01, .03, .05, .1, .25]
grid_max_depth = [2, 4, 6, 8, 10, None]
grid_min_samples = [5, 10, .03, .05, .10]
grid_criterion = ['gini', 'entropy']
grid_bool = [True, False]
grid_seed = [0]

grid_param = [
	# [{
	# 	# AdaBoostClassifier - http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html
	# 	'n_estimators': grid_n_estimator,  # default=50
	# 	'learning_rate': grid_learn,  # default=1
	# 	# 'algorithm': ['SAMME', 'SAMME.R'], #default=’SAMME.R
	# 	'random_state': grid_seed
	# }],
	#
	[{
		# BaggingClassifier - http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.BaggingClassifier.html#sklearn.ensemble.BaggingClassifier
		'n_estimators': grid_n_estimator,  # default=10
		'max_samples': grid_ratio,  # default=1.0
		'random_state': grid_seed
	}],
	#
	# [{
	# 	# ExtraTreesClassifier - http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesClassifier.html#sklearn.ensemble.ExtraTreesClassifier
	# 	'n_estimators': grid_n_estimator,  # default=10
	# 	'criterion': grid_criterion,  # default=”gini”
	# 	'max_depth': grid_max_depth,  # default=None
	# 	'random_state': grid_seed
	# }],
	#
	# [{
	# 	# GradientBoostingClassifier - http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html#sklearn.ensemble.GradientBoostingClassifier
	# 	# 'loss': ['deviance', 'exponential'], #default=’deviance’
	# 	'learning_rate': [.05],
	# 	# default=0.1 -- 12/31/17 set to reduce runtime -- The best parameter for GradientBoostingClassifier is
	# 	# {'learning_rate': 0.05, 'max_depth': 2, 'n_estimators': 300, 'random_state': 0} with a runtime of 264.45 seconds.
	# 	'n_estimators': [300],
	# 	# default=100 -- 12/31/17 set to reduce runtime -- The best parameter for GradientBoostingClassifier is
	# 	# {'learning_rate': 0.05, 'max_depth': 2, 'n_estimators': 300, 'random_state': 0} with a runtime of 264.45 seconds.
	# 	# 'criterion': ['friedman_mse', 'mse', 'mae'], #default=”friedman_mse”
	# 	'max_depth': grid_max_depth,  # default=3
	# 	'random_state': grid_seed
	# }],
	#
	# [{
	# 	# RandomForestClassifier - http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html#sklearn.ensemble.RandomForestClassifier
	# 	'n_estimators': grid_n_estimator,  # default=10
	# 	'criterion': grid_criterion,  # default=”gini”
	# 	'max_depth': grid_max_depth,  # default=None
	# 	'oob_score': [True],
	# 	# default=False -- 12/31/17 set to reduce runtime -- The best parameter for RandomForestClassifier is
	# 	# {'criterion': 'entropy', 'max_depth': 6, 'n_estimators': 100, 'oob_score': True, 'random_state': 0} with a runtime of 146.35 seconds.
	# 	'random_state': grid_seed
	# }],
	#
	# [{
	# 	# GaussianProcessClassifier
	# 	'max_iter_predict': grid_n_estimator,  # default: 100
	# 	'random_state': grid_seed
	# }],
	#
	# [{
	# 	# LogisticRegressionCV - http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegressionCV.html#sklearn.linear_model.LogisticRegressionCV
	# 	'fit_intercept': grid_bool,  # default: True
	# 	# 'penalty': ['l1','l2'],
	# 	'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],  # default: lbfgs
	# 	'random_state': grid_seed
	# }],
	#
	# [{
	# 	# BernoulliNB - http://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.BernoulliNB.html#sklearn.naive_bayes.BernoulliNB
	# 	'alpha': grid_ratio,  # default: 1.0
	# }],
	#
	# # GaussianNB -
	# [{}],
	#
	# [{
	# 	# KNeighborsClassifier - http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html#sklearn.neighbors.KNeighborsClassifier
	# 	'n_neighbors': [1, 2, 3, 4, 5, 6, 7],  # default: 5
	# 	'weights': ['uniform', 'distance'],  # default = ‘uniform’
	# 	'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
	# }],
	#
	# [{
	# 	# SVC - http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC
	# 	# http://blog.hackerearth.com/simple-tutorial-svm-parameter-tuning-python-r
	# 	# 'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
	# 	'C': [1, 2, 3, 4, 5],  # default=1.0
	# 	'gamma': grid_ratio,  # edfault: auto
	# 	'decision_function_shape': ['ovo', 'ovr'],  # default:ovr
	# 	'probability': [True],
	# 	'random_state': grid_seed
	# }],
	#
	# [{
	# 	# XGBClassifier - http://xgboost.readthedocs.io/en/latest/parameter.html
	# 	'learning_rate': grid_learn,  # default: .3
	# 	'max_depth': [1, 2, 4, 6, 8, 10],  # default 2
	# 	'n_estimators': grid_n_estimator,
	# 	'seed': grid_seed
	# }]
]
from datetime import datetime

start_total = time.perf_counter()  # https://docs.python.org/3/library/time.html#time.perf_counter
for clf, param in zip(vote_est, grid_param):  # https://docs.python.org/3/library/functions.html#zip

	# print(clf[1]) #vote_est is a list of tuples, index 0 is the name and index 1 is the algorithm
	# print(param)
	print(f'{datetime.now} clf: {clf} ; param:{param} start')
	start = time.perf_counter()
	best_search = model_selection.GridSearchCV(estimator=clf[1], param_grid=param, cv=cv_split, scoring='roc_auc')
	best_search.fit(train[data1_x_calc], train[Target])
	run = time.perf_counter() - start

	best_param = best_search.best_params_
	print('The best parameter for {} is {} with a runtime of {:.2f} seconds.'.format(clf[1].__class__.__name__,
																					 best_param, run))
	clf[1].set_params(**best_param)
	print(f'{datetime.now} clf: {clf} ; param:{param} end')

run_total = time.perf_counter() - start_total
print('Total optimization time was {:.2f} minutes.'.format(run_total / 60))

print('-' * 10)

# Hard Vote or majority rules w/Tuned Hyperparameters
grid_hard = ensemble.VotingClassifier(estimators=vote_est, voting='hard')
grid_hard_cv = model_selection.cross_validate(grid_hard, train[data1_x_calc], train[Target], cv=cv_split,
											  return_train_score=True)
grid_hard.fit(train[data1_x_calc], train[Target])

print("Hard Voting w/Tuned Hyperparameters Training w/bin score mean: {:.2f}".format(
	grid_hard_cv['train_score'].mean() * 100))
print(
	"Hard Voting w/Tuned Hyperparameters Test w/bin score mean: {:.2f}".format(grid_hard_cv['test_score'].mean() * 100))
print("Hard Voting w/Tuned Hyperparameters Test w/bin score 3*std: +/- {:.2f}".format(
	grid_hard_cv['test_score'].std() * 100 * 3))
print('-' * 10)

# Soft Vote or weighted probabilities w/Tuned Hyperparameters
grid_soft = ensemble.VotingClassifier(estimators=vote_est, voting='soft')
grid_soft_cv = model_selection.cross_validate(grid_soft, train[data1_x_calc], train[Target], cv=cv_split,
											  return_train_score=True)
grid_soft.fit(train[data1_x_calc], train[Target])

print("Soft Voting w/Tuned Hyperparameters Training w/bin score mean: {:.2f}".format(
	grid_soft_cv['train_score'].mean() * 100))
print(
	"Soft Voting w/Tuned Hyperparameters Test w/bin score mean: {:.2f}".format(grid_soft_cv['test_score'].mean() * 100))
print("Soft Voting w/Tuned Hyperparameters Test w/bin score 3*std: +/- {:.2f}".format(
	grid_soft_cv['test_score'].std() * 100 * 3))
print('-' * 10)
# prepare input for modeling
print(data_val.info())
print("-" * 10)
# data_val.sample(10)

# handmade decision tree - submission score = 0.77990
# data_val['Transported'] = mytree(data_val).astype(int)  # 0 V7
# data_val[Target] = mytree(data_val)

# decision tree w/full dataset modeling submission score: defaults= 0.76555, tuned= 0.77990
# submit_dt = tree.DecisionTreeClassifier()
# submit_dt = model_selection.GridSearchCV(tree.DecisionTreeClassifier(), param_grid=param_grid, scoring = 'roc_auc', cv = cv_split)
# submit_dt.fit(data1[data1_x_bin], data1[Target])
# print('Best Parameters: ', submit_dt.best_params_) #Best Parameters:  {'criterion': 'gini', 'max_depth': 4, 'random_state': 0}
# data_val['Survived'] = submit_dt.predict(data_val[data1_x_bin])


# bagging w/full dataset modeling submission score: defaults= 0.75119, tuned= 0.77990
# submit_bc = ensemble.BaggingClassifier()
# submit_bc = model_selection.GridSearchCV(ensemble.BaggingClassifier(), param_grid= {'n_estimators':grid_n_estimator, 'max_samples': grid_ratio, 'oob_score': grid_bool, 'random_state': grid_seed}, scoring = 'roc_auc', cv = cv_split)
# submit_bc.fit(data1[data1_x_bin], data1[Target])
# print('Best Parameters: ', submit_bc.best_params_) #Best Parameters:  {'max_samples': 0.25, 'n_estimators': 500, 'oob_score': True, 'random_state': 0}
# data_val['Survived'] = submit_bc.predict(data_val[data1_x_bin])


# extra tree w/full dataset modeling submission score: defaults= 0.76555, tuned= 0.77990
# submit_etc = ensemble.ExtraTreesClassifier()
# submit_etc = model_selection.GridSearchCV(ensemble.ExtraTreesClassifier(), param_grid={'n_estimators': grid_n_estimator, 'criterion': grid_criterion, 'max_depth': grid_max_depth, 'random_state': grid_seed}, scoring = 'roc_auc', cv = cv_split)
# submit_etc.fit(data1[data1_x_bin], data1[Target])
# print('Best Parameters: ', submit_etc.best_params_) #Best Parameters:  {'criterion': 'entropy', 'max_depth': 6, 'n_estimators': 100, 'random_state': 0}
# data_val['Survived'] = submit_etc.predict(data_val[data1_x_bin])


# random foreset w/full dataset modeling submission score: defaults= 0.71291, tuned= 0.73205
# submit_rfc = ensemble.RandomForestClassifier()
# submit_rfc = model_selection.GridSearchCV(ensemble.RandomForestClassifier(), param_grid={'n_estimators': grid_n_estimator, 'criterion': grid_criterion, 'max_depth': grid_max_depth, 'random_state': grid_seed}, scoring = 'roc_auc', cv = cv_split)
# submit_rfc.fit(data1[data1_x_bin], data1[Target])
# print('Best Parameters: ', submit_rfc.best_params_) #Best Parameters:  {'criterion': 'entropy', 'max_depth': 6, 'n_estimators': 100, 'random_state': 0}
# data_val['Survived'] = submit_rfc.predict(data_val[data1_x_bin])


# ada boosting w/full dataset modeling submission score: defaults= 0.74162, tuned= 0.75119
# submit_abc = ensemble.AdaBoostClassifier()
# submit_abc = model_selection.GridSearchCV(ensemble.AdaBoostClassifier(), param_grid={'n_estimators': grid_n_estimator, 'learning_rate': grid_ratio, 'algorithm': ['SAMME', 'SAMME.R'], 'random_state': grid_seed}, scoring = 'roc_auc', cv = cv_split)
# submit_abc.fit(data1[data1_x_bin], data1[Target])
# print('Best Parameters: ', submit_abc.best_params_) #Best Parameters:  {'algorithm': 'SAMME.R', 'learning_rate': 0.1, 'n_estimators': 300, 'random_state': 0}
# data_val['Survived'] = submit_abc.predict(data_val[data1_x_bin])


# gradient boosting w/full dataset modeling submission score: defaults= 0.75119, tuned= 0.77033
# submit_gbc = ensemble.GradientBoostingClassifier()
# submit_gbc = model_selection.GridSearchCV(ensemble.GradientBoostingClassifier(), param_grid={'learning_rate': grid_ratio, 'n_estimators': grid_n_estimator, 'max_depth': grid_max_depth, 'random_state':grid_seed}, scoring = 'roc_auc', cv = cv_split)
# submit_gbc.fit(data1[data1_x_bin], data1[Target])
# print('Best Parameters: ', submit_gbc.best_params_) #Best Parameters:  {'learning_rate': 0.25, 'max_depth': 2, 'n_estimators': 50, 'random_state': 0}
# data_val['Survived'] = submit_gbc.predict(data_val[data1_x_bin])

# extreme boosting w/full dataset modeling submission score: defaults= 0.73684, tuned= 0.77990
# submit_xgb = XGBClassifier()
# submit_xgb = model_selection.GridSearchCV(XGBClassifier(), param_grid= {'learning_rate': grid_learn, 'max_depth': [0,2,4,6,8,10], 'n_estimators': grid_n_estimator, 'seed': grid_seed}, scoring = 'roc_auc', cv = cv_split)
# submit_xgb.fit(data1[data1_x_bin], data1[Target])
# print('Best Parameters: ', submit_xgb.best_params_) #Best Parameters:  {'learning_rate': 0.01, 'max_depth': 4, 'n_estimators': 300, 'seed': 0}
# data_val['Survived'] = submit_xgb.predict(data_val[data1_x_bin])


# hard voting classifier w/full dataset modeling submission score: defaults=-, tuned = 0.74655 V4
# data_val[Target] = vote_hard.predict(data_val[data1_x_calc])  # 0.74655 V4
# data_val[Target] = grid_hard.predict(data_val[data1_x_calc])  # 0.70189 V4

# soft voting classifier w/full dataset modeling submission score: defaults=-, tuned = 0.75005 V6
# data_val[Target] = vote_soft.predict(data_val[data1_x_calc])  # 0.75005 V6
data_val[Target] = grid_soft.predict(data_val[data1_x_calc])  # 0.74982 V5
data_val[Target] = labelEncoder.inverse_transform(data_val[Target])

# submit file
tmpname = ''
for name in vote_est:
	tmpname = '_' + name[0]

submit = data_val[['id', Target]]
submitName = f"submission07_sum_{tmpname}.csv"
submit.to_csv(submitName, index=False)

print('Validation Data Distribution: \n', data_val[Target].value_counts(normalize=True))
submit.sample(10)

# The best parameter for AdaBoostClassifier is {'learning_rate': 0.1, 'n_estimators': 300, 'random_state': 0} with a runtime of 147.10 seconds.
# The best parameter for BaggingClassifier is {'max_samples': 0.1, 'n_estimators': 300, 'random_state': 0} with a runtime of 259.10 seconds.
# The best parameter for ExtraTreesClassifier is {'criterion': 'gini', 'max_depth': 8, 'n_estimators': 300, 'random_state': 0} with a runtime of 268.24 seconds.
# The best parameter for GradientBoostingClassifier is {'learning_rate': 0.05, 'max_depth': 4, 'n_estimators': 300, 'random_state': 0} with a runtime of 484.47 seconds.
# The best parameter for RandomForestClassifier is {'criterion': 'gini', 'max_depth': 8, 'n_estimators': 300, 'oob_score': True, 'random_state': 0} with a runtime of 376.14 seconds.
# The best parameter for GaussianProcessClassifier is {'max_iter_predict': 10, 'random_state': 0} with a runtime of 1475.33 seconds.
# The best parameter for LogisticRegressionCV is {'fit_intercept': True, 'random_state': 0, 'solver': 'lbfgs'} with a runtime of 433.93 seconds.
# The best parameter for BernoulliNB is {'alpha': 0.5} with a runtime of 1.08 seconds.
# The best parameter for GaussianNB is {} with a runtime of 0.25 seconds.
# The best parameter for KNeighborsClassifier is {'algorithm': 'ball_tree', 'n_neighbors': 7, 'weights': 'distance'} with a runtime of 46.03 seconds.
# The best parameter for SVC is {'C': 1, 'decision_function_shape': 'ovo', 'gamma': 0.1, 'probability': True, 'random_state': 0} with a runtime of 5311.01 seconds.
# The best parameter for XGBClassifier is {'learning_rate': 0.1, 'max_depth': 4, 'n_estimators': 50, 'seed': 0} with a runtime of 597.68 seconds.
# Total optimization time was 156.67 minutes.
# TODO

print(f'{datetime.now()}  {submitName} end !!!!!')
