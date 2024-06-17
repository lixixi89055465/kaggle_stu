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

train = reduce_mem_usage(train)
data_val = reduce_mem_usage(data_val)
print('0' * 100)


def getMean(df, col):
	train[f'{col}_mean'] = (train[col] - train[col].min()) / train[col].min()


print('3' * 100)
getMean(train, 'Admissiongrade')

data1_x_calc = ['Maritalstatus', 'Applicationmode', 'Applicationorder',
				'Course',  # ('Course', 0.0009095661265027921)
				'Daytimeeveningattendance', 'Previousqualification',
				'Previousqualificationgrade',  # ('Previousqualificationgrade', 0.0002090956612650352)
				# 'Nacionality',# ('Nacionality', 0.0003659174072138116)
				'Mothersqualification',
				'Fathersqualification',
				'Mothersoccupation', 'Fathersoccupation',
				'Admissiongrade',
				'Displaced',  # ('Displaced', 0.0002090956612650352)
				# 'Educationalspecialneeds', #('Educationalspecialneeds', 0.000261369576581294)
				'Debtor',
				'Tuitionfeesuptodate',
				'Gender',
				'Scholarshipholder',
				'Ageatenrollment',  # ('Ageatenrollment', 0.00019864087820178344)
				# 'International',#('International', 0.00019864087820178344)
				'Curricularunits1stsemcredited',
				'Curricularunits1stsemenrolled', 'Curricularunits1stsemevaluations',
				'Curricularunits1stsemapproved',
				'Curricularunits1stsemgrade',  # ('Curricularunits1stsemgrade', 0.0009618400418190509)
				'Curricularunits1stsemwithoutevaluations',
				# ('Curricularunits1stsemwithoutevaluations', 0.0008259278619967781)
				'Curricularunits2ndsemcredited', 'Curricularunits2ndsemenrolled',
				'Curricularunits2ndsemevaluations', 'Curricularunits2ndsemapproved',
				'Curricularunits2ndsemgrade', 'Curricularunits2ndsemwithoutevaluations',
				# ('Curricularunits2ndsemgrade', 0.0009200209095660439)
				'Unemploymentrate',
				'Inflationrate', 'GDP']
# cur_sum = 'curr_sum'

# cur_col = ['Curricularunits1stsemcredited',
# 		   'Curricularunits1stsemenrolled', 'Curricularunits1stsemevaluations',
# 		   'Curricularunits1stsemapproved', 'Curricularunits1stsemgrade',
# 		   'Curricularunits1stsemwithoutevaluations',
# 		   'Curricularunits2ndsemcredited', 'Curricularunits2ndsemenrolled',
# 		   'Curricularunits2ndsemevaluations', 'Curricularunits2ndsemapproved',
# 		   'Curricularunits2ndsemgrade', 'Curricularunits2ndsemwithoutevaluations' ]
# train[cur_sum] = train[cur_col].sum(axis=1)
# data1_x_calc += [cur_sum]
Target = 'Target'
labelEncoder = LabelEncoder()
train[Target] = labelEncoder.fit_transform(train[Target])
import eli5
from eli5.sklearn import PermutationImportance
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

train_X, val_X, train_y, val_y = train_test_split(train[data1_x_calc], train[Target], random_state=1)

my_model = RandomForestClassifier(n_estimators=30, random_state=1, n_jobs=-1).fit(train[data1_x_calc], train[Target])
perms = PermutationImportance(my_model, random_state=1).fit(val_X, val_y)
r1 = eli5.show_weights(perms, feature_names=val_X.columns.tolist())
print('3' * 100)
print(r1)

print('4' * 100)
r2 = eli5.show_weights(my_model)

feature_importances = perms.feature_importances_
m = {}
print('5' * 100)
for f, w in zip(data1_x_calc, feature_importances):
	m[f] = w
r = sorted(m.items(), key=lambda k: -k[1])
for i in r:
	print(i)

print(train['Curricularunits2ndsemgrade'].dtype == 'float16')
print(train['Curricularunits2ndsemgrade'].dtype)
# print('train.value_counts():')
# print(train.value_counts())
cols = list()
for col in data1_x_calc:
	if train[col].dtype in [np.int8, np.int16, np.int32, np.int64]:
		cols.append(col)
		print(f'{col} value count is : \t {train[col].value_counts()}')

import matplotlib.pyplot as plt

fig = plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
sns.histplot(data=train, x='Curricularunits2ndsemwithoutevaluations', hue=Target,
			 # bins=200
			 )
plt.title('Total expenditure (truncated)')
plt.ylim([0, 200])
plt.xlim([0, 20000])

plt.subplot(1, 2, 2)
sns.countplot(data=train, x='Curricularunits2ndsemwithoutevaluations', hue=Target)
plt.title('No spending indicator')
fig.tight_layout()
