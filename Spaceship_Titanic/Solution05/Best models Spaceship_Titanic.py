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

data_raw = pd.read_csv('../input/train.csv')
data_val = pd.read_csv('../input/test.csv')
# #to play with our input we'll create a copy
# #remember python assignment or equal passes by reference vs values, so we use the copy function: https://stackoverflow.com/questions/46327494/python-pandas-dataframe-copydeep-false-vs-copydeep-true-vs
data1 = data_raw.copy(deep=True)
data_cleaner = [data1, data_val]
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

print('Train columns with null value:\n', data1.isnull().sum())
print("-" * 100)
print('Test/Validation columns with null value :\n', data_val.isnull().sum())
print("- " * 100)
print(data_raw.nunique())
print(data_raw.dtypes)
# Expenditure features
exp_feats = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
# Categorical feature
cat_feats = ['HomePlanet', 'CryoSleep', 'Destination', 'VIP']
# Qualitative features
qual_feats = ['PassengerId', 'Cabin', 'Name']

for dataset in data_cleaner:
	dataset['Age_group'] = np.nan
	dataset.loc[dataset['Age'] <= 12, 'Age_group'] = 'Age_0-12'
	dataset.loc[(dataset['Age'] > 12) & (dataset['Age'] < 18), 'Age_group'] = 'Age_13-17'
	dataset.loc[(dataset['Age'] >= 18) & (dataset['Age'] <= 25), 'Age_group'] = 'Age_18-25'
	dataset.loc[(dataset['Age'] > 25) & (dataset['Age'] <= 30), 'Age_group'] = 'Age_26-30'
	dataset.loc[(dataset['Age'] > 30) & (dataset['Age'] <= 50), 'Age_group'] = 'Age_31-50'
	dataset.loc[dataset['Age'] > 50, 'Age_group'] = 'Age_51+'

# Plot distribution of new features
# plt.figure(figsize=(10, 4))
# g = sns.countplot(input=data1, x='Age_group', hue='Transported',
#                   order=['Age_0-12', 'Age_13-17', 'Age_18-25', 'Age_26-30', 'Age_31-50', 'Age_51+'])
# plt.title('Age group distribution')
# plt.show()

for dataset in data_cleaner:
	dataset['Expenditure'] = dataset[exp_feats].sum(axis=1)
	dataset['No_spending'] = (dataset['Expenditure'] == 0).astype(int)

# fig = plt.figure(figsize=(12, 4))
# plt.subplot(1, 2, 1)
# sns.histplot(input=data1, x='Expenditure', hue='Transported', bins=200)
# plt.title('Total expenditure (truncated) ')
# plt.ylim([0, 200])
# plt.xlim([0, 20000])
# plt.subplot(1, 2, 2)
# sns.countplot(input=data1, x='No_spending', hue='Transported')
# plt.title('No spending indicator')
# fig.tight_layout()
# plt.show()
# print('0'*100)

# Plot distribution of new features
fig = plt.figure(figsize=(12, 4))
# plt.subplot(1, 2, 1)
# sns.histplot(input=data1, x='Expenditure', hue='Transported', bins=200)
# plt.title('Total expenditure (truncated)')
# plt.ylim([0, 200])
# plt.xlim([0, 20000])
#
# plt.subplot(1, 2, 2)
# sns.countplot(input=data1, x='No_spending', hue='Transported')
# plt.title('No spending indicator')
# fig.tight_layout()
for dataset in data_cleaner:
	dataset['Group'] = dataset['PassengerId'].apply(lambda x: x.split('_')[0]).astype(int)
	dvc = dataset['Group'].value_counts()
	dataset['Group_size'] = dataset['Group'].map(lambda x: dvc[x])
# dataset['Solo'] = (dataset['Group_size'] == 1).astype(int)

# plt.figure(figsize=(20, 16))
# plt.subplot(1, 2, 1)
# sns.histplot(input=data1, x='Group', hue='Transported', binwidth=1)
# plt.title('Group')
#
# plt.subplot(1, 2, 2)
# sns.countplot(input=data1, x='Group_size', hue='Transported')
# plt.title('Group size')
# fig.tight_layout()
# plt.show()

# print("0" * 100)
## New features
# data_raw['Solo'] = (data_raw['Group_size'] == 1).astype(int)
# data_val['Solo'] = (data_val['Group_size'] == 1).astype(int)

# print('1' * 100)
for dataset in data_cleaner:
	dataset['Solo'] = (dataset['Group_size'] == 1).astype(int)
# New feature distribution
# plt.figure(figsize=(10, 4))
# sns.countplot(input=data1, x='Solo', hue='Transported')
# plt.title('Passenger travelling sole  or not ')
# plt.ylim([0, 3000])
# plt.show()

for dataset in data_cleaner:
	# Replace NaN's with outliers for now (so we can split feature)
	dataset['Cabin'].fillna('Z/9999/Z', inplace=True)
	# New features
	dataset['Cabin_deck'] = dataset['Cabin'].apply(lambda x: x.split('/')[0])
	dataset['Cabin_number'] = dataset['Cabin'].apply(lambda x: x.split('/')[1]).astype(int)
	dataset['Cabin_side'] = dataset['Cabin'].apply(lambda x: x.split('/')[2])

	# Put Nan's back in (we will fill these later)
	dataset.loc[dataset['Cabin_deck'] == 'Z', 'Cabin_deck'] = np.nan
	dataset.loc[dataset['Cabin_number'] == 9999, 'Cabin_number'] = np.nan
	dataset.loc[dataset['Cabin_side'] == 'Z', 'Cabin_side'] = np.nan

	# Drop Cabin (we don't need it anymore)
	dataset.drop('Cabin', axis=1, inplace=True)
# plot distribution of new features
# fig = plt.figure(figsize=(10, 12))
# plt.subplot(3, 1, 1)
# sns.countplot(input=data1, x='Cabin_deck', hue='Transported', order=['A', 'B', 'C', 'D', 'E', 'F', 'G', 'T'])
# plt.title('Cabin-deck')
#
# plt.subplot(3, 1, 2)
# sns.histplot(input=data1, x='Cabin_number', hue='Transported', binwidth=20)
# plt.vlines(300, ymin=0, ymax=200, color='black')
# plt.vlines(600, ymin=0, ymax=200, color='black')
# plt.vlines(900, ymin=0, ymax=200, color='black')
# plt.vlines(1200, ymin=0, ymax=200, color='black')
# plt.vlines(1500, ymin=0, ymax=200, color='black')
# plt.vlines(1800, ymin=0, ymax=200, color='black')
# plt.title('Cabin number')
# plt.xlim([0, 2000])
#
# plt.subplot(3, 1, 3)
# sns.countplot(input=data1, x='Cabin_side', hue='Transported')
# plt.title('cabin side')
# fig.tight_layout()
# plt.show()
# print("5" * 100)
for dataset in data_cleaner:
	# New feature - training set
	dataset['Cabin_region1'] = (dataset['Cabin_number'] < 300).astype(int)
	dataset['Cabin_region2'] = ((dataset['Cabin_number'] >= 300) & (dataset['Cabin_number'] < 600)).astype(int)
	dataset['Cabin_region3'] = ((dataset['Cabin_number'] >= 600) & (dataset['Cabin_number'] < 900)).astype(int)
	dataset['Cabin_region4'] = ((dataset['Cabin_number'] >= 900) & (dataset['Cabin_number'] < 1200)).astype(int)
	dataset['Cabin_region5'] = ((dataset['Cabin_number'] >= 1200) & (dataset['Cabin_number'] < 1500)).astype(int)
	dataset['Cabin_region6'] = ((dataset['Cabin_number'] >= 1500) & (dataset['Cabin_number'] < 1500)).astype(int)
	dataset['Cabin_region7'] = (dataset['Cabin_number'] >= 1800).astype(int)

# plt.figure(figsize=(10, 4))
data1['Cabin_region_plot'] = (
		data1['Cabin_region1'] + 2 * data1['Cabin_region2'] + 3 * data1['Cabin_region3'] + 4 * data1[
	'Cabin_region4'] + 5 * data1['Cabin_region5'] + 6 * data1['Cabin_region6'] + 7 * data1['Cabin_region7']).astype(int)
# sns.countplot(input=data1, x='Cabin_region_plot', hue='Transported')
# plt.title('Cabin region ')
# plt.show()
# data1.drop('Cabin_region_plot', axis=1, inplace=True)
#
for dataset in data_cleaner:
	dataset['Name'].fillna('Unknown Unknown', inplace=True)
	dataset['Surname'] = dataset['Name'].str.split().str[-1]
	dataset['Family_size'] = dataset['Surname'].map(lambda x: dataset['Surname'].value_counts()[x])
	testSurname = dataset['Surname'].value_counts()
	# dataset['Surname'].map(lambda x: dataset['Surname'].value_counts()[x])
	dataset['Surname'].map(lambda x: testSurname[x])
	# dataset.loc[dataset['Surname'] == 'Unknow Unknow'] = np.nan
	dataset.loc[dataset['Surname'] == 'Unknown', 'Surname'] = np.nan
	dataset.loc[dataset['Family_size'] > 100, 'Family_size'] = np.nan
	dataset.drop('Name', axis=1, inplace=True)

# New  feature distribution
#
# plt.figure(figsize=(12, 4))
# sns.countplot(input=data1, x='Family_size', hue='Transported')
# plt.title('Family_size ')
# plt.show()
# Missing values¶

# data1['Transported'].astype(int)
for dataset in data_cleaner:
	# Columns with missing values
	na_cols = dataset.columns[dataset.isna().any()].tolist()
	mv = pd.DataFrame(dataset[na_cols].isna().sum(), columns=['Number_missing'])
	mv['Percentage_missing'] = np.round(100 * mv['Number_missing'] / len(dataset), 2)
	print(mv, '\n')
data1['na_count'] = data1.isna().sum(axis=1)

# Countplot of number of missing values by passenger
# plt.figure(figsize=(10, 4))
# sns.countplot(input=data1, x='na_count', hue='Transported')
# plt.title("number of missing entries by pasenger")
# plt.show()
# data1.drop('na_count', axis=1, inplace=True)

# We managed to fill 131 values with 100% confidence but we are nott finished yet.


# TODO
for dataset in data_cleaner:
	GHP_gb = dataset.groupby(['Group', 'HomePlanet'])['HomePlanet'].size().unstack().fillna(0)
	# Missing values before
	HP_bef = dataset['HomePlanet'].isna().sum()
	GHP_index = dataset[dataset['HomePlanet'].isna()][
		(dataset[dataset['HomePlanet'].isna()]['Group']).isin(GHP_gb.index)].index
	dataset.loc[GHP_index, 'HomePlanet'] = dataset.iloc[GHP_index, :]['Group'].map(lambda x: GHP_gb.idxmax(axis=1)[x])
	print('#Missing values before:', HP_bef)
	print("#Missing values after:", dataset['HomePlanet'].isna().sum())

print('5' * 100)
for dataset in data_cleaner:
	HP_bef = dataset['HomePlanet'].isna().sum()
	dataset.loc[
		(dataset['HomePlanet'].isna()) & (dataset['Cabin_deck'].isin(['A', 'B', 'C', 'T'])), 'HomePlanet'] = 'Europa'
	dataset.loc[(dataset['HomePlanet'].isna()) & (dataset['Cabin_deck'] == 'G'), 'HomePlanet'] = 'Earth'
# print('#HomePlanet missing values before:', HP_bef)
# print('#HomePlanet missing values after:', dataset['HomePlanet'].isna().sum())

print('6' * 100)
for data in data_cleaner:
	SHP_gb = data.groupby(['Surname', 'HomePlanet'])['HomePlanet'].size().unstack().fillna(0)
	HP_bef = data['HomePlanet'].isna().sum()
	SHP_index = data[data['HomePlanet'].isna()][(data[data['HomePlanet'].isna()]['Surname']).isin(SHP_gb.index)].index
	data.loc[SHP_index, 'HomePlanet'] = data.iloc[SHP_index, :]['Surname'].map(lambda x: SHP_gb.idxmax(axis=1)[x])

# Print number of missing values left
# print('#HomePlanet missing values before:', HP_bef)
# print('#HomePlanet missing values after:', input['HomePlanet'].isna().sum())

print('7' * 100)
# TODO
for data in data_cleaner:
	HP_bef = data['HomePlanet'].isna().sum()
	data.loc[(data['HomePlanet'].isna()) & ~(data['Cabin_deck'] == 'D'), 'HomePlanet'] = 'Earth'
	data.loc[(data['HomePlanet'].isna()) & (data['Cabin_deck'] == 'D'), 'HomePlanet'] = 'Mars'
# print('#HomePlanet missing values before:', HP_bef)
# print('#HomePlanet missing values after:', input['HomePlanet'].isna().sum())
# We're done with HomePlanet.
print('8' * 100)
for data in data_cleaner:
	D_bef = data['Destination'].isna().sum()
	data.loc[data['Destination'].isna(), 'Destination'] = 'TRAPPIST-1e'

print('9' * 100)
for data in data_cleaner:
	GSN_gb = data[data['Group_size'] > 1].groupby(['Group', 'Surname'])['Surname'].size().unstack().fillna(0)
	# GSN_gb.sum(axis=1)
	SN_bef = data['Surname'].isna().sum()
	GSN_index = data[data['Surname'].isna()][(data[data['Surname'].isna()]['Group']).isin(GSN_gb.index)].index
	# Fill corresponding missing values
	data.loc[GSN_index, 'Surname'] = data.iloc[GSN_index, :]['Group'].map(lambda x: GSN_gb.idxmax(axis=1)[x])
	# Print number of missing values left
	# print('#Surname missing values before:', SN_bef)
	# print('#Surname missing values after:', input['Surname'].isna().sum())
	data['Surname'].fillna('Unknown', inplace=True)
	data['Family_size'] = data['Surname'].map(lambda x: data['Surname'].value_counts()[x])
	data.loc[data['Surname'] == 'Unknown', 'Surname'] = np.nan
	data.loc[data['Family_size'] > 100, 'Family_size'] = 0

print('0' * 100)
for data in data_cleaner:
	GCD_gb = data[data['Group_size'] > 1].groupby(['Group', 'Cabin_deck'])['Cabin_deck'].size().unstack().fillna(0)
	GCN_gb = data[data['Group_size'] > 1].groupby(['Group', 'Cabin_number'])['Cabin_number'].size().unstack().fillna(0)
	GCS_gb = data[data['Group_size'] > 1].groupby(['Group', 'Cabin_side'])['Cabin_side'].size().unstack().fillna(0)
	# Everyone in the same group is also on the same cabin side. For cabin deck and cabin number there is also a fairly good (but not perfect) correlation with group.
	# input[input['Group_size'] > 1].groupby(['Group', 'Cabin_side'])['Group'].size().unstack().fillna(0)
	# Missing values before
	CS_bef = data['Cabin_side'].isna().sum()
	GCS_index = data[data['Cabin_side'].isna()][(data[data['Cabin_side'].isna()]['Group']).isin(GCS_gb.index)].index
	data.loc[GCS_index, 'Cabin_side'] = data.iloc[GCS_index, :]['Group'].map(lambda x: GCS_gb.idxmax(axis=1)[x])
# Print number of missing values left
# print('#Cabin_side missing values before:', CS_bef)
# print('#Cabin_side missing values after:', input['Cabin_side'].isna().sum())
print('1' * 100)  # TODO
for data in data_cleaner:
	# Joint distribution of Surname and Cabin side
	SCS_gb = data[data['Group_size'] > 1].groupby(['Surname', 'Cabin_side'])['Cabin_side'].size().unstack().fillna(0)
	# Ratio of sides
	SCS_gb['Ratio'] = SCS_gb['P'] / (SCS_gb['P'] + SCS_gb['S'])
# Histograme of ratio
# plt.figure(figsize=(10, 4))
# sns.histplot(SCS_gb['Ratio'], kde=True, binwidth=0.05)
# plt.title('Ratio of cabin side by surname')
# plt.show()
# Print proportion
# TODO
print("Percentage of families all on the same cabin side:", 100 * np.round(SCS_gb['Ratio'] / len(SCS_gb), 3), '%')
print("2" * 100)
print(SCS_gb.head())

# Missing values before
CS_bef = data['Cabin_deck'].isna().sum()
print("3" * 100)
print(CS_bef)
# Drop ratio columns
SCS_gb.drop(['Ratio'], axis=1, inplace=True)
for data in data_cleaner:
	SCS_index = data[data['Cabin_deck'].isna()][(data[data['Cabin_deck'].isna()]['Surname']).isin(SCS_gb.index)].index
	data.loc[SCS_index, 'Cabin_deck'] = data.iloc[SCS_index, :]['Surname'].map(lambda x: SCS_gb.idxmax(axis=1)[x])
	data.drop('Surname', axis=1, inplace=True)
	print('#Cabin side missing before:', CS_bef)
	print('#Cabin side missing after:', data['Cabin_side'].isna().sum())
# TODO
print("3" * 100)
for data in data_cleaner:
	# Value counts
	print(data['Cabin_side'].value_counts())
	# Missing values before
	CS_bef = data['Cabin_side'].isna().sum()
	data.loc[data['Cabin_side'].isna(), 'Cabin_deck'] = 'Z'
# print('#Cabin side missing values before:', CS_bef)
# print('#Cabin side missing value after:', input['Cabin_side'].isna().sum())

print("4" * 100)
for data in data_cleaner:
	# print(input['Cabin_deck'].value_counts())
	CD_bef = data['Cabin_deck'].isna().sum()
	GCD_index = data[data['Cabin_deck'].isna()][(data[data['Cabin_deck'].isna()]['Group']).
		isin(GCD_gb.index)].index
	# Fill corresponding missing values
	data.loc[GCD_index, 'Cabin_deck'] = data.iloc[GCD_index, :]['Group']. \
		map(lambda x: GCD_gb.idxmax(axis=1)[x])
# Print number of missing values left
# print('#Cabin_deck missing values before:', CD_bef)
# print('#Cabin_deck missing values after:', input['Cabin_deck'].isna().sum())

print("7" * 100)
# TODO
for data in data_cleaner:
	# Missing values before
	CD_bef = data['Cabin_deck'].isna().sum()

	# Passengers with missing Cabin deck and in a group with known majority Cabin deck
	GCD_index = data[data['Cabin_deck'].isna()][(data[data['Cabin_deck'].isna()]['Group']).isin(GCD_gb.index)].index

	# Fill corresponding missing values
	data.loc[GCD_index, 'Cabin_deck'] = data.iloc[GCD_index, :]['Group'].map(lambda x: GCD_gb.idxmax(axis=1)[x])

	# Print number of missing values left
	print('#Cabin_deck missing values before:', CD_bef)
	print('#Cabin_deck missing values after:', data['Cabin_deck'].isna().sum())

for data in data_cleaner:
	# Joint distribution
	data.groupby(['HomePlanet', 'Destination', 'Solo', 'Cabin_deck'])['Cabin_deck'].size().unstack().fillna(0)
	'''
	Passengers from Mars are most likely in deck F.
	Passengers from Europa are (more or less) most likely in deck C if travelling solo and deck B otherwise.
	Passengers from Earth are (more or less) most likely in deck G.
	We will fill in missing values according to where the mode appears in these subgroups.
	'''
	# Missing values before
	CD_bef = data['Cabin_deck'].isna().sum()

	# Fill missing values using the mode
	na_rows_CD = data.loc[data['Cabin_deck'].isna(), 'Cabin_deck'].index
	print(len(na_rows_CD))
	data.loc[data['Cabin_deck'].isna(), 'Cabin_deck'] = data.groupby(
		['HomePlanet', 'Destination', 'Solo'])['Cabin_deck'].transform(
		lambda x: x.fillna(pd.Series.mode(x)[0]))[na_rows_CD]

	# Print number of missing values left
	print('#Cabin_deck missing values before:', CD_bef)
	print('#Cabin_deck missing values after:', data['Cabin_deck'].isna().sum())
print("8" * 100)
# for input in data_cleaner:
#     plt.figure(figsize=(10, 4))
#     sns.scatterplot(x=input['Cabin_number'], y=input['Group'],
#                     c=LabelEncoder().fit_transform(input.loc[~input['Cabin_number'].isna(), 'Cabin_deck']), cmap='tab10')
#     plt.title('Cabin_number vs group colored by group')
# plt.show()
print('9' * 100)

from sklearn.linear_model import LinearRegression

for data in data_cleaner:
	# Missing values before
	CN_bef = data['Cabin_number'].isna().sum()
	print('#Cabin_number missing values before:', CN_bef)
	# Extrapolate linear relationship on a deck by deck basis
	for deck in ['A', 'B', 'C', 'D', 'E', 'F', 'G']:
		# Features and labels
		X_CN = data.loc[~(data['Cabin_number'].isna()) & (data['Cabin_deck'] == deck), 'Group']
		y_CN = data.loc[~(data['Cabin_number'].isna()) & (data['Cabin_deck'] == deck), 'Cabin_number']
		X_test_CN = data.loc[(data['Cabin_number'].isna()) & (data['Cabin_deck'] == deck), 'Group']

		if not X_test_CN.empty:
			# Linear regression
			# model_CN = sklearn.linear_model.LinearRegression()
			model_CN = LinearRegression()
			model_CN.fit(X_CN.values.reshape(-1, 1), y_CN)
			preds_CN = model_CN.predict(X_test_CN.values.reshape(-1, 1))
			# Fill missing values with predictions
			data.loc[(data['Cabin_number'].isna()) & (data['Cabin_deck'] == deck), 'Cabin_number'] = preds_CN.astype(
				int)

# Print number of missing values left
print('#Cabin_number missing values before:', CN_bef)
print('#Cabin_number missing values after:', data['Cabin_number'].isna().sum())

# One-hot encode cabin regions
data['Cabin_region1'] = (data['Cabin_number'] < 300).astype(int)
data['Cabin_region2'] = ((data['Cabin_number'] >= 300) & (data['Cabin_number'] < 600)).astype(int)
data['Cabin_region3'] = ((data['Cabin_number'] >= 600) & (data['Cabin_number'] < 900)).astype(int)
data['Cabin_region4'] = ((data['Cabin_number'] >= 900) & (data['Cabin_number'] < 1200)).astype(int)
data['Cabin_region5'] = ((data['Cabin_number'] >= 1200) & (data['Cabin_number'] < 1500)).astype(int)
data['Cabin_region6'] = ((data['Cabin_number'] >= 1500) & (data['Cabin_number'] < 1800)).astype(int)
data['Cabin_region7'] = (data['Cabin_number'] >= 1800).astype(int)
print('0' * 100)
for data in data_cleaner:
	# Missing values before
	V_bef = data['VIP'].isna().sum()
	# # Fill missing values with mode
	data.loc[data['VIP'].isna(), 'VIP'] = False
# print('#VIP missing values before:', V_bef)
# print('#VIP missing values after:', input['VIP'].isna().sum())
# print('1' * 100)
for data in data_cleaner:
	data.groupby(['HomePlanet', 'No_spending', 'Solo', 'Cabin_deck'])['Age'].median().unstack().fillna(0)
	# Missing values before
	A_bef = data[exp_feats].isna().sum().sum()
	# # Fill missing values using the median
	# na_rows_A = input.loc[input['Age'].isna(), 'Age'].index
	# input.loc[input['Age'].isna(), 'Age'] = \
	#     input.groupby(['HomePlanet', 'No_spending', 'Solo', 'Cabin_deck'])['Age'].transform(
	#         lambda x: x.fillna(x.median()))[na_rows_A]
	na_rows_A = data.loc[data['Age'].isna(), 'Age'].index
	# TODO
	data.loc[data['Age'].isna(), 'Age'] = \
		data.groupby(['HomePlanet', 'No_spending', 'Solo', 'Cabin_deck'])['Age'].transform(
			lambda x: x.fillna(x.median()))[
			na_rows_A]
# # Print number of missing values left
# print('#Age missing values before:', A_bef)
# input.groupby(['HomePlanet', 'No_spending', 'Solo', 'Cabin_deck'])['Age'].transform(lambda x: x.fillna(x.median()))[
#     na_rows_A]

print('2' * 100)
for data in data_cleaner:
	data.loc[data['Age'] <= 12, 'Age_group'] = 'Age_0-12'
	data.loc[(data['Age'] > 12) & (data['Age'] < 18), 'Age_group'] = 'Age_13-17'
	data.loc[(data['Age'] >= 18) & (data['Age'] <= 25), 'Age_group'] = 'Age_18-25'
	data.loc[(data['Age'] > 25) & (data['Age'] <= 30), 'Age_group'] = 'Age_26-30'
	data.loc[(data['Age'] > 30) & (data['Age'] <= 50), 'Age_group'] = 'Age_31-50'
	data.loc[data['Age'] > 50, 'Agegroup'] = 'Age_51+'

for data in data_cleaner:
	# Join distribution
	data.groupby(['No_spending', 'CryoSleep'])['CryoSleep'].size().unstack().fillna(0)
	CSL_bef = data['CryoSleep'].isna().sum()
	# # Fill missing values using the mode
	na_rows_CSL = data.loc[data['CryoSleep'].isna(), 'CryoSleep'].index
	data.loc[data['CryoSleep'].isna(), 'CryoSleep'] = \
		data.groupby(['No_spending'])['CryoSleep'].transform(lambda x: x.fillna(pd.Series.mode(x)[0]))[na_rows_CSL]
# print("#CryoSleep misisng values before:", CSL_bef)
# print('#CryoSleep missing values after:', input['CryoSleep'].isna().sum())

# TODO
for data in data_cleaner:
	# Missing value before
	E_bef = data[exp_feats].isna().sum().sum()
	for col in exp_feats:
		data.loc[(data[col].isna()) & (data['CryoSleep'] == True), col] = 0
# print("#Expenditure missing values before", E_bef)
# print("#Expenditure missing values after", input[exp_feats].isna().sum().sum())
print('-' * 100)

for data in data_cleaner:
	data.groupby(['HomePlanet', 'Solo', 'Age_group'])['Expenditure'].mean().unstack().fillna(0)
	E_bef = data[exp_feats].isna().sum().sum()
	for col in exp_feats:
		na_rows = data.loc[data[col].isna(), col].index
		data.loc[data[col].isna(), col] = \
			data.groupby(['HomePlanet', 'Solo', 'Age_group'])[col].transform(lambda x: x.fillna(x.mean()))[na_rows]
		print('#Expenditure missing values before:', E_bef)
		print('#Expendigure missing values after:', data[exp_feats].isna().sum().sum())

for data in data_cleaner:
	data['Expenditure'] = data[exp_feats].sum(axis=1)
	data['No_spending'] = (data['Expenditure'] == 0).astype(int)
	# input.isna().sum()
	for col in ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'Expenditure']:
		data[col] = np.log(1 + data[col])

## 3.23 Convert Formats
'''
## 3.23 Convert Formats

We will convert categorical input to dummy variables for mathematical analysis.
 There are multiple ways to encode categorical variables;we will use the sklearn and pandas functions.

In this step, we will also define our x (independent/features/explanatory/predictor/etc.) 
and y (dependent/target/outcome/response/etc.) variables for input modeling.

** Developer Documentation: **
* [Categorical Encoding](http://pbpython.com/categorical-encoding.html)
* [Sklearn LabelEncoder](http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html)
* [Sklearn OneHotEncoder](http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html)
* [Pandas Categorical dtype](https://pandas.pydata.org/pandas-docs/stable/categorical.html)
* [pandas.get_dummies](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.get_dummies.html)
'''
###TODO 待定
label = LabelEncoder()
for data in data_cleaner:
	data['HomePlanet_Code'] = label.fit_transform(data['HomePlanet'])
	data['CryoSleep_Code'] = label.fit_transform(data['CryoSleep'])
	data['Destination_Code'] = label.fit_transform(data['Destination'])
	data['VIP_Code'] = label.fit_transform(data['VIP'])
	data['Age_group_Code'] = label.fit_transform(data['Age_group'])
	data['Cabin_deck_Code'] = label.fit_transform(data['Cabin_deck'])
	data['Cabin_side_Code'] = label.fit_transform(data['Cabin_side'])

Target = ['Transported']
data1_x = ['HomePlanet', 'CryoSleep',
		   'Destination', 'Age', 'VIP',
		   'RoomService', 'FoodCourt',
		   'ShoppingMall', 'Spa',
		   'VRDeck']  # Original input
data1_x_calc = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'Expenditure', 'No_spending',
				'Group', 'Group_size', 'Solo', 'Cabin_number', 'Cabin_region1', 'Cabin_region2', 'Cabin_region3',
				'Cabin_region4', 'Cabin_region5', 'Cabin_region6', 'Cabin_region7', 'Family_size', 'HomePlanet_Code',
				'CryoSleep_Code', 'Destination_Code', 'VIP_Code', 'Age_group_Code', 'Cabin_deck_Code',
				'Cabin_side_Code']  # coded for algorithm calculation
data1_xy = Target + data1_x
print('Original X Y:', data1_xy, '\n')
# defina x variables for original w/bin features to remove continuous variables
data1_x_bin = ['Age', 'No_spending', 'Group_size', 'Solo', 'Cabin_region1',
			   'Cabin_region2', 'Cabin_region3',
			   'Cabin_region4', 'Cabin_region5',
			   'Cabin_region6', 'Cabin_region7',
			   'Family_size', 'HomePlanet_Code',
			   'CryoSleep_Code', 'Destination_Code',
			   'VIP_Code', 'Age_group_Code',
			   'Cabin_deck_Code', 'Cabin_side_Code']
data1_xy_bin = Target + data1_x_bin
print('Bin X Y: ', data1_xy_bin, '\n')
# define x and y variables for dummy features original
data1_dummy = pd.get_dummies(data1[data1_x])
data1_x_dummy = data1_dummy.columns.to_list()
data1_xy_dummy = Target + data1_x_dummy
print("Dummy X Y:", data1_xy_dummy, '\n')
print(data1_dummy.head())

# 3.24 Da-Double Check Cleaned Data
print('Train columns with null values: \n', data1.isnull().sum())
print("-" * 10)
print(data1.info())
print("-" * 10)

print('Test/Validation columns with null values: \n', data_val.isnull().sum())
print("-" * 10)
print(data_val.info())
print("-" * 10)
from sklearn import model_selection

print(data_raw.describe(include='all'))
train_x, test1_x, train1_y, test1_y =model_selection.train_test_split(data1[data1_x_calc], data1[Target],
																	   random_state=0)

train_x_bin, test1_x_bin, train1_y_bin, test1_y_bin = model_selection.train_test_split(data1[data1_x_bin],
																					   data1[Target], random_state=0)

train1_x_dummy, test1_x_dummy, train1_y_dummy, test1_y_dummy = model_selection.train_test_split(
	data1_dummy[data1_x_dummy], data1[Target], random_state=0)


# print("Data1 Shape: {}".format(data1.shape))
# print("Train1 Shape: {}".format(train1_x.shape))
# print("Test1 Shape: {}".format(test1_x.shape))
# print(train1_x_dummy.head())

# for x in data1_x:
# 	if data1[x].dtype != 'float64':
# 		print('Transported Correlation by :', x)
# 		print(data1[[x, Target[0]]].groupby(x, as_index=False).mean())
# 		print('-' * 10, '\n')


# correlation heatmap of dataset
def correlation_heatmap(df):
	_, ax = plt.subplots(figsize=(14, 12))
	colormap = sns.diverging_palette(220, 10, as_cmap=True)
	_ = sns.heatmap(
		df.corr(),
		cmap=colormap,
		square=True,
		ax=ax,
		annot=True,
		linewidths=0.1, vmax=1.0, linecolor='white',
		annot_kws={'fontsize': 5}
	)


# 	plt.title('Pearson correlation of features', y=1.05, size=15)


# correlation_heatmap(data1)

print('6' * 100)
# pair plots of entire dataset
# pp = sns.pairplot(data1, hue='Transported', palette='deep', size=1.2, diag_kind='kde', diag_kws=dict(shade=True),
# 				  plot_kws=dict(s=10))
# pp.set(xticklabels=[])
# plt.show()

# Machine Learning Algorithm (MLA) Selection and Initialization
from sklearn import ensemble, gaussian_process, \
	linear_model, naive_bayes, \
	neighbors, svm, tree, discriminant_analysis
from xgboost import XGBClassifier

MLA = [
	# Ensemble Methods
	ensemble.AdaBoostClassifier(),
	ensemble.BaggingClassifier(),
	# ensemble.ExtraTreesClassifier(),
	# # TODO
	# ensemble.GradientBoostingClassifier(),
	# ensemble.RandomForestClassifier(),
	#
	# # Gaussian Processes
	# gaussian_process.GaussianProcessClassifier(),
	# # GLM
	# linear_model.LogisticRegressionCV(),
	# linear_model.PassiveAggressiveClassifier(),
	# linear_model.RidgeClassifierCV(),
	# linear_model.SGDClassifier(),
	# linear_model.Perceptron(),
	#
	# # Navies Bayer
	# naive_bayes.BernoulliNB(),
	# naive_bayes.GaussianNB(),
	# # Nearest Neighbor
	# neighbors.KNeighborsClassifier(),
	#
	# # SVM
	# svm.SVC(probability=True),
	# svm.NuSVC(probability=True),
	# svm.LinearSVC(),
	# # Trees
	# tree.DecisionTreeClassifier(),
	# tree.ExtraTreeClassifier(),
	# # Discriminat Analysisi
	# discriminant_analysis.LinearDiscriminantAnalysis(),
	# discriminant_analysis.QuadraticDiscriminantAnalysis(),
	# #xgboost: http://xgboost.readthedocs.io/en/latest/model.html
	# # TODO
	# XGBClassifier()
]
# split dataset in cross-validation with this splitter class:
# http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.ShuffleSplit.html#sklearn.model_selection.ShuffleSplit
# note: this is an alternative to train_test_split
cv_split = model_selection.ShuffleSplit(n_splits=10, test_size=.3, train_size=.6, random_state=0)
# create table to compare MLA metrics
MLA_columns = ['MLA Name', 'MLA Parameters', 'MLA Train Accuracy Mean', \
			   'MLA Test Accuracy Mean', 'MLA Test Accuracy 3*STD', 'MLA Time']
MLA_compare = pd.DataFrame(columns=MLA_columns)

# create table to compare MLA predictions
MLA_predict = data1[Target]

# index through MLA and save performance to table
row_index = 0
for alg in MLA:
	MLA_name = alg.__class__.__name__
	MLA_compare.loc[row_index, 'MLA Name'] = MLA_name
	MLA_compare.loc[row_index, 'MLA Parameters'] = str(alg.get_params())
	# score model with cross validation: http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_validate.html#sklearn.model_selection.cross_validate
	cv_results = model_selection.cross_validate(
		alg,  #
		data1[data1_x_bin],  #
		data1[Target],  #
		cv=cv_split,  #
		return_train_score=True
	)
	MLA_compare.loc[row_index, 'MLA Time'] = cv_results['fit_time'].mean()
	MLA_compare.loc[row_index, 'MLA Train Accuracy Mean'] = cv_results['train_score'].mean()
	MLA_compare.loc[row_index, 'MLA Test Accuracy Mean'] = cv_results['test_score'].mean()
	# if this is a non-bias random sample, then +/-3 standard deviations (std) from the mean, should statistically capture 99.7% of the subsets
	MLA_compare.loc[row_index, 'MLA Test accuracy 3*STD'] = cv_results['test_score'].std() * 3

	# Save MLA predictions - see section 6 for usage
	alg.fit(data1[data1_x_bin], data1[Target])
	MLA_predict[MLA_name] = alg.predict(data1[data1_x_bin])
	row_index += 1

# print and sort table: https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.sort_values.html
MLA_compare.sort_values(by=['MLA Test Accuracy Mean'], ascending=False, inplace=True)
# MLA_compare
# barplot using https://seaborn.pydata.org/generated/seaborn.barplot.html
# sns.barplot(x='MLA test Accuracy Mean', y='MLA name ', input=MLA_compare, color='m')
# # prettify using pyplot: https://matplotlib.org/api/pyplot_api.html
# plt.title('Machine Learning Algorithm Accuracy Score \n')
# plt.xlabel('Accuracy Score ( % )')
# plt.ylabel('Algorithm')

for index, row in data1.iterrows():
	if random.random() > .5:
		data1.at[index, 'Random_Predict'] = 1
	else:
		data1.at[index, 'Random_Predict'] = 0

# score random guess of survival. Use shortcut 1 = Right Guess and 0 = Wrong Guess
# the mean of the column will then equal the accuracy
data1['Random_Score'] = 0
data1.loc[(data1['Transported'] == data1['Random_Predict']), 'Random_Score'] = 1
print('Coin Flip Model Accuracy :{:.2f}%'.format(data1['Random_Score'].mean() * 100))

print('Coin Flip Model Accuracy w/Scikit:{:.2f}%'.format(metrics.accuracy_score(data1[Target],  #
																				data1['Random_Predict']) * 100))

print(data1.info())

pivot_age = data1.groupby(['Age_group'])['Transported'].mean()
print('Survival Decision Tree age Node: \n', pivot_age)

pivot_vip = data1.groupby(['VIP_Code'])['Transported'].mean()
print('\n survival Decision Tree Vip node:\n', pivot_vip)

pivot_HomePlanet = data1.groupby(['HomePlanet_Code'])['Transported'].mean()
print('\n survival Decision Tree HomePlanet node:\n', pivot_vip)

pivot_CryoSleep = data1.groupby(['CryoSleep_Code'])['Transported'].mean()
print('\n Survival Decision Tree CryoSleep Node:\n', pivot_CryoSleep)

pivot_Cabin_deck_Code = data1.groupby(['Cabin_deck_Code'])['Transported'].mean()
print('\n Survival Decision Tree CryoSleep Node:\n', pivot_CryoSleep)

pivot_Cabin_side_Code = data1.groupby(['Cabin_side_Code'])['Transported'].mean()
print('\n Survival Decision Tree Cabin_side Code:\n', pivot_CryoSleep)


# pivot_male = data1[data1.Sex=='male'].groupby(['Sex','Title'])['Transported'].mean()
# print('\n\nSurvival Decision Tree w/Male Node: \n',pivot_male)

# handmade input model using brain power (and Microsoft Excel Pivot Tables for quick calculations
def mytree(df):
	# initialize table to store predictions
	Model = pd.DataFrame(data={'Predict': []})
	for index, row in df.iterrows():
		# Question 1 :Age group (55=69%)
		if (df.loc[index, 'Age_group'] == 'Age_0-12') or \
				(df.loc[index, 'Age_group'] == 'Age_13-17'):
			Model.loc[index, 'Predict'] = 1
		# Question 2:HomePlanet_code (66-67%)
		if (df.loc[index, 'HomePlanet_Code'] == 0):
			Model.loc[index, 'Predict'] = 0
		if (df.loc[index, 'HomePlanet_Code'] == 1):
			Model.loc[index, 'Predict'] = 1
		# Question 3:vip_code (71%)
		if (df.loc[index, 'VIP_Code'] == 1):
			Model.loc[index, 'Predict'] = 0
		# Question 4: CryoSleep_Code (68-81%)
		if (df.loc[index, 'CryoSleep_Code'] == 0):
			Model.loc[index, 'Predict'] = 0
		if (df.loc[index, 'CryoSleep_Code'] == 1):
			Model.loc[index, 'Predict'] = 1
		# Question 5: Cabin_deck_Code_Code (73-80%)
		if (df.loc[index, 'Cabin_deck_Code'] == 7):
			Model.loc[index, 'Predict'] = 0
		if (df.loc[index, 'Cabin_deck_Code'] == 1):
			Model.loc[index, 'Predict'] = 1
		if (df.loc[index, 'Cabin_deck_Code'] == 2):
			Model.loc[index, 'Predict'] = 1
		if (df.loc[index, 'Cabin_deck_Code'] == 4):
			Model.loc[index, 'Predict'] = 0
		# Question 6: Cabin_side_Code (72%)
		if (df.loc[index, 'Cabin_side_Code'] == 2):
			Model.loc[index, 'Predict'] = 0
	return Model


# model input
Tree_Predict = mytree(data1)

# TODO
print('Decision Tree Model Accuracy /Precision Score:{:.2f}%\n'.
	  format(metrics.accuracy_score(data1['Transported'], Tree_Predict) * 100))
# Accuracy Summary Report with http://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html#sklearn.metrics.classification_report
# Where recall score = (true positives)/(true positive + false negative) w/1 being best:http://scikit-learn.org/stable/modules/generated/sklearn.metrics.recall_score.html#sklearn.metrics.recall_score
# And F1 score = weighted average of precision and recall w/1 being best: http://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html#sklearn.metrics.f1_score
print(metrics.classification_report(data1['Transported'], Tree_Predict))
# Plot Accuracy Summary
# Credit: http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
import itertools


def plot_confusion_matrix(cm, classes,
						  normalize=False,
						  title='Confusion matrix',
						  cmap=plt.cm.Blues):
	"""
	This function prints and plots the confusion matrix.
	Normalization can be applied by setting `normalize=True`.
	"""
	if normalize:
		cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
		print("Normalized confusion matrix")
	else:
		print('Confusion matrix, without normalization')

	print(cm)

	plt.imshow(cm, interpolation='nearest', cmap=cmap)
	plt.title(title)
	plt.colorbar()
	tick_marks = np.arange(len(classes))
	plt.xticks(tick_marks, classes, rotation=45)
	plt.yticks(tick_marks, classes)

	fmt = '.2f' if normalize else 'd'
	thresh = cm.max() / 2.
	for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
		plt.text(j, i, format(cm[i, j], fmt),
				 horizontalalignment="center",
				 color="white" if cm[i, j] > thresh else "black")

	plt.tight_layout()
	plt.ylabel('True label')
	plt.xlabel('Predicted label')


# Compute confusion matrix
cnf_matrix = metrics.confusion_matrix(data1['Transported'], Tree_Predict)
np.set_printoptions(precision=2)

class_names = ['NotTransported', 'Transported']
# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names,
					  title='Confusion matrix, without normalization')

# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
					  title='Normalized confusion matrix')

# base model
dtree = tree.DecisionTreeClassifier(random_state=0)
base_results = model_selection.cross_validate(dtree, data1[data1_x_bin], data1[Target], cv=cv_split,
											  return_train_score=True)
dtree.fit(data1[data1_x_bin], data1[Target])

print('BEFORE DT Parameters: ', dtree.get_params())
print("BEFORE DT Training w/bin score mean: {:.2f}".format(base_results['train_score'].mean() * 100))
print("BEFORE DT Test w/bin score mean: {:.2f}".format(base_results['test_score'].mean() * 100))
print("BEFORE DT Test w/bin score 3*std: +/- {:.2f}".format(base_results['test_score'].std() * 100 * 3))
# print("BEFORE DT Test w/bin set score min: {:.2f}". format(base_results['test_score'].min()*100))
print('-' * 10)

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
tune_model = model_selection.GridSearchCV(tree.DecisionTreeClassifier(), param_grid=param_grid, scoring='roc_auc',
										  cv=cv_split, return_train_score=True)
tune_model.fit(data1[data1_x_bin], data1[Target])

# print(tune_model.cv_results_.keys())
# print(tune_model.cv_results_['params'])
print('AFTER DT Parameters: ', tune_model.best_params_)
# print(tune_model.cv_results_['mean_train_score'])
print("AFTER DT Training w/bin score mean: {:.2f}".format(
	tune_model.cv_results_['mean_train_score'][tune_model.best_index_] * 100))
# print(tune_model.cv_results_['mean_test_score'])
print("AFTER DT Test w/bin score mean: {:.2f}".format(
	tune_model.cv_results_['mean_test_score'][tune_model.best_index_] * 100))
print("AFTER DT Test w/bin score 3*std: +/- {:.2f}".format(
	tune_model.cv_results_['std_test_score'][tune_model.best_index_] * 100 * 3))
print('-' * 10)

# duplicates gridsearchcv
# tune_results = model_selection.cross_validate(tune_model, data1[data1_x_bin], data1[Target], cv  = cv_split)

# print('AFTER DT Parameters: ', tune_model.best_params_)
# print("AFTER DT Training w/bin set score mean: {:.2f}". format(tune_results['train_score'].mean()*100))
# print("AFTER DT Test w/bin set score mean: {:.2f}". format(tune_results['test_score'].mean()*100))
# print("AFTER DT Test w/bin set score min: {:.2f}". format(tune_results['test_score'].min()*100))
# print('-'*10)


# base model
print('BEFORE DT RFE Training Shape Old: ', data1[data1_x_bin].shape)
print('BEFORE DT RFE Training Columns Old: ', data1[data1_x_bin].columns.values)

print("BEFORE DT RFE Training w/bin score mean: {:.2f}".format(base_results['train_score'].mean() * 100))
print("BEFORE DT RFE Test w/bin score mean: {:.2f}".format(base_results['test_score'].mean() * 100))
print("BEFORE DT RFE Test w/bin score 3*std: +/- {:.2f}".format(base_results['test_score'].std() * 100 * 3))
print('-' * 10)

#TODO
from sklearn import feature_selection
#TODO
# feature selection
dtree_rfe = feature_selection.RFECV(dtree, step=1, scoring='accuracy', cv=cv_split)
dtree_rfe.fit(data1[data1_x_bin], data1[Target])

# transform x&y to reduced features and fit new model
# alternative: can use pipeline to reduce fit and transform steps:
# http://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html
X_rfe = data1[data1_x_bin].columns.values[dtree_rfe.get_support()]
rfe_results = model_selection.cross_validate(dtree, \
											 data1[X_rfe], \
											 data1[Target], \
											 cv=cv_split, \
											 return_train_score=True)
# print(dtree_rfe.grid_scores_)
print('AFTER DT RFE Training Shape New: ', data1[X_rfe].shape)
print('AFTER DT RFE Training Columns New: ', X_rfe)

print("AFTER DT RFE Training w/bin score mean: {:.2f}".format(rfe_results['train_score'].mean() * 100))
print("AFTER DT RFE Test w/bin score mean: {:.2f}".format(rfe_results['test_score'].mean() * 100))
print("AFTER DT RFE Test w/bin score 3*std: +/- {:.2f}".format(rfe_results['test_score'].std() * 100 * 3))
print('-' * 10)

# tune rfe model
rfe_tune_model = model_selection.GridSearchCV(tree.DecisionTreeClassifier(),\
											  param_grid=param_grid,\
											  scoring='roc_auc',\
											  cv=cv_split,\
											  return_train_score=True)
rfe_tune_model.fit(data1[X_rfe], data1[Target])

# print(rfe_tune_model.cv_results_.keys())
# print(rfe_tune_model.cv_results_['params'])
print('AFTER DT RFE Tuned Parameters: ', rfe_tune_model.best_params_)
# print(rfe_tune_model.cv_results_['mean_train_score'])
print("AFTER DT RFE Tuned Training w/bin score mean: {:.2f}".format(
	rfe_tune_model.cv_results_['mean_train_score'][tune_model.best_index_] * 100))
# print(rfe_tune_model.cv_results_['mean_test_score'])
print("AFTER DT RFE Tuned Test w/bin score mean: {:.2f}".format(
	rfe_tune_model.cv_results_['mean_test_score'][tune_model.best_index_] * 100))
print("AFTER DT RFE Tuned Test w/bin score 3*std: +/- {:.2f}".format(
	rfe_tune_model.cv_results_['std_test_score'][tune_model.best_index_] * 100 * 3))
print('-' * 10)

# Graph MLA version of Decision Tree: http://scikit-learn.org/stable/modules/generated/sklearn.tree.export_graphviz.html
import graphviz

dot_data = tree.export_graphviz(dtree, out_file=None,
								feature_names=data1_x_bin, class_names=True,
								filled=True, rounded=True)
graph = graphviz.Source(dot_data)
graph

# compare algorithm predictions with each other, where 1 = exactly similar and 0 = exactly opposite
# there are some 1's, but enough blues and light reds to create a "super algorithm" by combining them
correlation_heatmap(MLA_predict)
# why choose one model, when you can pick them all with voting classifier
# http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.VotingClassifier.html
# removed models w/o attribute 'predict_proba' required for
# vote classifier and models with a 1.0 correlation to
# another model
from sklearn.svm import SVC

vote_est = [
	# Ensemble Methods: http://scikit-learn.org/stable/modules/ensemble.html
	('ada', ensemble.AdaBoostClassifier()),
	('bc', ensemble.BaggingClassifier()),
	# ('etc', ensemble.ExtraTreesClassifier()),
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
vote_hard_cv = model_selection.cross_validate(vote_hard, data1[data1_x_bin], data1[Target], \
											  cv=cv_split, \
											  return_train_score=True)
vote_hard.fit(data1[data1_x_bin], data1[Target])
print("Hard Voting Training w/bin score mean: {:.2f}".format(vote_hard_cv['train_score'].mean() * 100))
print("Hard Voting Test w/bin score mean: {:.2f}".format(vote_hard_cv['test_score'].mean() * 100))
print("Hard Voting Test w/bin score 3*std: +/- {:.2f}".format(vote_hard_cv['test_score'].std() * 100 * 3))
print('-' * 10)

# Soft Vote or weighted probabilities

vote_soft = ensemble.VotingClassifier(estimators=vote_est, voting='soft')
vote_soft_cv = model_selection.cross_validate(vote_soft, data1[data1_x_bin], data1[Target], \
											  cv=cv_split, return_train_score=True)
vote_soft.fit(data1[data1_x_bin], data1[Target])

print("Soft Voting Training w/bin score mean: {:.2f}".format(vote_soft_cv['train_score'].mean() * 100))
print("Soft Voting Test w/bin score mean: {:.2f}".format(vote_soft_cv['test_score'].mean() * 100))
print("Soft Voting Test w/bin score 3*std: +/- {:.2f}".format(vote_soft_cv['test_score'].std() * 100 * 3))
print('-' * 10)
# IMPORTANT: THIS SECTION IS UNDER CONSTRUCTION!!!!
# UPDATE: This section was scrapped for the next section; as it's more computational friendly.

# WARNING: Running is very computational intensive and time expensive code is written for experimental/developmental purposes and not production ready


# tune each estimator before creating a super model
# http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html
grid_n_estimator = [50,100,300]
grid_ratio = [.1,.25,.5,.75,1.0]
grid_learn = [.01,.03,.05,.1,.25]
grid_max_depth = [2,4,6,None]
grid_min_samples = [5,10,.03,.05,.10]
grid_criterion = ['gini', 'entropy']
grid_bool = [True, False]
grid_seed = [0]

vote_param = [{
	#            #http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html
	'ada__n_estimators': grid_n_estimator,
	'ada__learning_rate': grid_ratio,
	'ada__algorithm': ['SAMME', 'SAMME.R'],
	'ada__random_state': grid_seed,

	# http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.BaggingClassifier.html#sklearn.ensemble.BaggingClassifier
	'bc__n_estimators': grid_n_estimator,
	'bc__max_samples': grid_ratio,
	'bc__oob_score': grid_bool,
	'bc__random_state': grid_seed,

	# http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesClassifier.html#sklearn.ensemble.ExtraTreesClassifier
	'etc__n_estimators': grid_n_estimator,
	'etc__criterion': grid_criterion,
	'etc__max_depth': grid_max_depth,
	'etc__random_state': grid_seed,

	# http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html#sklearn.ensemble.GradientBoostingClassifier
	'gbc__loss': ['deviance', 'exponential'],
	'gbc__learning_rate': grid_ratio,
	'gbc__n_estimators': grid_n_estimator,
	'gbc__criterion': ['friedman_mse', 'mse', 'mae'],
	'gbc__max_depth': grid_max_depth,
	'gbc__min_samples_split': grid_min_samples,
	'gbc__min_samples_leaf': grid_min_samples,
	'gbc__random_state': grid_seed,

	# http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html#sklearn.ensemble.RandomForestClassifier
	'rfc__n_estimators': grid_n_estimator,
	'rfc__criterion': grid_criterion,
	'rfc__max_depth': grid_max_depth,
	'rfc__min_samples_split': grid_min_samples,
	'rfc__min_samples_leaf': grid_min_samples,
	'rfc__bootstrap': grid_bool,
	'rfc__oob_score': grid_bool,
	'rfc__random_state': grid_seed,

	# http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegressionCV.html#sklearn.linear_model.LogisticRegressionCV
	'lr__fit_intercept': grid_bool,
	'lr__penalty': ['l1', 'l2'],
	'lr__solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
	'lr__random_state': grid_seed,

	# http://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.BernoulliNB.html#sklearn.naive_bayes.BernoulliNB
	'bnb__alpha': grid_ratio,
	'bnb__prior': grid_bool,
	'bnb__random_state': grid_seed,

	# http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html#sklearn.neighbors.KNeighborsClassifier
	'knn__n_neighbors': [1, 2, 3, 4, 5, 6, 7],
	'knn__weights': ['uniform', 'distance'],
	'knn__algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
	'knn__random_state': grid_seed,

	# http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC
	# http://blog.hackerearth.com/simple-tutorial-svm-parameter-tuning-python-r
	'svc__kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
	'svc__C': grid_max_depth,
	'svc__gamma': grid_ratio,
	'svc__decision_function_shape': ['ovo', 'ovr'],
	'svc__probability': [True],
	'svc__random_state': grid_seed,

	# http://xgboost.readthedocs.io/en/latest/parameter.html
	'xgb__learning_rate': grid_ratio,
	'xgb__max_depth': [2, 4, 6, 8, 10],
	'xgb__tree_method': ['exact', 'approx', 'hist'],
	'xgb__objective': ['reg:linear', 'reg:logistic', 'binary:logistic'],
	'xgb__seed': grid_seed

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
	[{
		# AdaBoostClassifier - http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html
		'n_estimators': grid_n_estimator,  # default=50
		'learning_rate': grid_learn,  # default=1
		# 'algorithm': ['SAMME', 'SAMME.R'], #default=’SAMME.R
		'random_state': grid_seed
	}],

	[{
		# BaggingClassifier - http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.BaggingClassifier.html#sklearn.ensemble.BaggingClassifier
		'n_estimators': grid_n_estimator,  # default=10
		'max_samples': grid_ratio,  # default=1.0
		'random_state': grid_seed
	}],

	[{
		# ExtraTreesClassifier - http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesClassifier.html#sklearn.ensemble.ExtraTreesClassifier
		'n_estimators': grid_n_estimator,  # default=10
		'criterion': grid_criterion,  # default=”gini”
		'max_depth': grid_max_depth,  # default=None
		'random_state': grid_seed
	}],

	[{
		# GradientBoostingClassifier - http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html#sklearn.ensemble.GradientBoostingClassifier
		# 'loss': ['deviance', 'exponential'], #default=’deviance’
		'learning_rate': [.05],
		# default=0.1 -- 12/31/17 set to reduce runtime -- The best parameter for GradientBoostingClassifier is
		# {'learning_rate': 0.05, 'max_depth': 2, 'n_estimators': 300, 'random_state': 0} with a runtime of 264.45 seconds.
		'n_estimators': [300],
		# default=100 -- 12/31/17 set to reduce runtime -- The best parameter for GradientBoostingClassifier is
		# {'learning_rate': 0.05, 'max_depth': 2, 'n_estimators': 300, 'random_state': 0} with a runtime of 264.45 seconds.
		# 'criterion': ['friedman_mse', 'mse', 'mae'], #default=”friedman_mse”
		'max_depth': grid_max_depth,  # default=3
		'random_state': grid_seed
	}],

	[{
		# RandomForestClassifier - http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html#sklearn.ensemble.RandomForestClassifier
		'n_estimators': grid_n_estimator,  # default=10
		'criterion': grid_criterion,  # default=”gini”
		'max_depth': grid_max_depth,  # default=None
		'oob_score': [True],
		# default=False -- 12/31/17 set to reduce runtime -- The best parameter for RandomForestClassifier is
		# {'criterion': 'entropy', 'max_depth': 6, 'n_estimators': 100, 'oob_score': True, 'random_state': 0} with a runtime of 146.35 seconds.
		'random_state': grid_seed
	}],

	[{
		# GaussianProcessClassifier
		'max_iter_predict': grid_n_estimator,  # default: 100
		'random_state': grid_seed
	}],

	[{
		# LogisticRegressionCV - http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegressionCV.html#sklearn.linear_model.LogisticRegressionCV
		'fit_intercept': grid_bool,  # default: True
		# 'penalty': ['l1','l2'],
		'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],  # default: lbfgs
		'random_state': grid_seed
	}],

	[{
		# BernoulliNB - http://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.BernoulliNB.html#sklearn.naive_bayes.BernoulliNB
		'alpha': grid_ratio,  # default: 1.0
	}],

	# GaussianNB -
	[{}],

	[{
		# KNeighborsClassifier - http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html#sklearn.neighbors.KNeighborsClassifier
		'n_neighbors': [1, 2, 3, 4, 5, 6, 7],  # default: 5
		'weights': ['uniform', 'distance'],  # default = ‘uniform’
		'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
	}],

	[{
		# SVC - http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC
		# http://blog.hackerearth.com/simple-tutorial-svm-parameter-tuning-python-r
		# 'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
		'C': [1, 2, 3, 4, 5],  # default=1.0
		'gamma': grid_ratio,  # edfault: auto
		'decision_function_shape': ['ovo', 'ovr'],  # default:ovr
		'probability': [True],
		'random_state': grid_seed
	}],

	[{
		# XGBClassifier - http://xgboost.readthedocs.io/en/latest/parameter.html
		'learning_rate': grid_learn,  # default: .3
		'max_depth': [1, 2, 4, 6, 8, 10],  # default 2
		'n_estimators': grid_n_estimator,
		'seed': grid_seed
	}]
]

start_total = time.perf_counter()  # https://docs.python.org/3/library/time.html#time.perf_counter
for clf, param in zip(vote_est, grid_param):  # https://docs.python.org/3/library/functions.html#zip

	# print(clf[1]) #vote_est is a list of tuples, index 0 is the name and index 1 is the algorithm
	# print(param)

	start = time.perf_counter()
	best_search = model_selection.GridSearchCV(estimator=clf[1], param_grid=param, cv=cv_split, scoring='roc_auc')
	best_search.fit(data1[data1_x_bin], data1[Target])
	run = time.perf_counter() - start

	best_param = best_search.best_params_
	print('The best parameter for {} is {} with a runtime of {:.2f} seconds.'.format(clf[1].__class__.__name__,
																					 best_param, run))
	clf[1].set_params(**best_param)

run_total = time.perf_counter() - start_total
print('Total optimization time was {:.2f} minutes.'.format(run_total / 60))

print('-' * 10)


#Hard Vote or majority rules w/Tuned Hyperparameters
grid_hard = ensemble.VotingClassifier(estimators = vote_est , voting = 'hard')
grid_hard_cv = model_selection.cross_validate(grid_hard, data1[data1_x_bin], data1[Target], cv  = cv_split, return_train_score=True)
grid_hard.fit(data1[data1_x_bin], data1[Target])

print("Hard Voting w/Tuned Hyperparameters Training w/bin score mean: {:.2f}". format(grid_hard_cv['train_score'].mean()*100))
print("Hard Voting w/Tuned Hyperparameters Test w/bin score mean: {:.2f}". format(grid_hard_cv['test_score'].mean()*100))
print("Hard Voting w/Tuned Hyperparameters Test w/bin score 3*std: +/- {:.2f}". format(grid_hard_cv['test_score'].std()*100*3))
print('-'*10)

#Soft Vote or weighted probabilities w/Tuned Hyperparameters
grid_soft = ensemble.VotingClassifier(estimators = vote_est , voting = 'soft')
grid_soft_cv = model_selection.cross_validate(grid_soft, data1[data1_x_bin], data1[Target], cv  = cv_split, return_train_score=True)
grid_soft.fit(data1[data1_x_bin], data1[Target])

print("Soft Voting w/Tuned Hyperparameters Training w/bin score mean: {:.2f}". format(grid_soft_cv['train_score'].mean()*100))
print("Soft Voting w/Tuned Hyperparameters Test w/bin score mean: {:.2f}". format(grid_soft_cv['test_score'].mean()*100))
print("Soft Voting w/Tuned Hyperparameters Test w/bin score 3*std: +/- {:.2f}". format(grid_soft_cv['test_score'].std()*100*3))
print('-'*10)
#prepare input for modeling
print(data_val.info())
print("-"*10)
#data_val.sample(10)

#handmade decision tree - submission score = 0.77990
# data_val['Transported'] = mytree(data_val).astype(int)  # 0 V7
data_val['Transported'] = mytree(data_val)

#decision tree w/full dataset modeling submission score: defaults= 0.76555, tuned= 0.77990
#submit_dt = tree.DecisionTreeClassifier()
#submit_dt = model_selection.GridSearchCV(tree.DecisionTreeClassifier(), param_grid=param_grid, scoring = 'roc_auc', cv = cv_split)
#submit_dt.fit(data1[data1_x_bin], data1[Target])
#print('Best Parameters: ', submit_dt.best_params_) #Best Parameters:  {'criterion': 'gini', 'max_depth': 4, 'random_state': 0}
#data_val['Survived'] = submit_dt.predict(data_val[data1_x_bin])


#bagging w/full dataset modeling submission score: defaults= 0.75119, tuned= 0.77990
#submit_bc = ensemble.BaggingClassifier()
#submit_bc = model_selection.GridSearchCV(ensemble.BaggingClassifier(), param_grid= {'n_estimators':grid_n_estimator, 'max_samples': grid_ratio, 'oob_score': grid_bool, 'random_state': grid_seed}, scoring = 'roc_auc', cv = cv_split)
#submit_bc.fit(data1[data1_x_bin], data1[Target])
#print('Best Parameters: ', submit_bc.best_params_) #Best Parameters:  {'max_samples': 0.25, 'n_estimators': 500, 'oob_score': True, 'random_state': 0}
#data_val['Survived'] = submit_bc.predict(data_val[data1_x_bin])


#extra tree w/full dataset modeling submission score: defaults= 0.76555, tuned= 0.77990
#submit_etc = ensemble.ExtraTreesClassifier()
#submit_etc = model_selection.GridSearchCV(ensemble.ExtraTreesClassifier(), param_grid={'n_estimators': grid_n_estimator, 'criterion': grid_criterion, 'max_depth': grid_max_depth, 'random_state': grid_seed}, scoring = 'roc_auc', cv = cv_split)
#submit_etc.fit(data1[data1_x_bin], data1[Target])
#print('Best Parameters: ', submit_etc.best_params_) #Best Parameters:  {'criterion': 'entropy', 'max_depth': 6, 'n_estimators': 100, 'random_state': 0}
#data_val['Survived'] = submit_etc.predict(data_val[data1_x_bin])


#random foreset w/full dataset modeling submission score: defaults= 0.71291, tuned= 0.73205
#submit_rfc = ensemble.RandomForestClassifier()
#submit_rfc = model_selection.GridSearchCV(ensemble.RandomForestClassifier(), param_grid={'n_estimators': grid_n_estimator, 'criterion': grid_criterion, 'max_depth': grid_max_depth, 'random_state': grid_seed}, scoring = 'roc_auc', cv = cv_split)
#submit_rfc.fit(data1[data1_x_bin], data1[Target])
#print('Best Parameters: ', submit_rfc.best_params_) #Best Parameters:  {'criterion': 'entropy', 'max_depth': 6, 'n_estimators': 100, 'random_state': 0}
#data_val['Survived'] = submit_rfc.predict(data_val[data1_x_bin])


#ada boosting w/full dataset modeling submission score: defaults= 0.74162, tuned= 0.75119
#submit_abc = ensemble.AdaBoostClassifier()
#submit_abc = model_selection.GridSearchCV(ensemble.AdaBoostClassifier(), param_grid={'n_estimators': grid_n_estimator, 'learning_rate': grid_ratio, 'algorithm': ['SAMME', 'SAMME.R'], 'random_state': grid_seed}, scoring = 'roc_auc', cv = cv_split)
#submit_abc.fit(data1[data1_x_bin], data1[Target])
#print('Best Parameters: ', submit_abc.best_params_) #Best Parameters:  {'algorithm': 'SAMME.R', 'learning_rate': 0.1, 'n_estimators': 300, 'random_state': 0}
#data_val['Survived'] = submit_abc.predict(data_val[data1_x_bin])


#gradient boosting w/full dataset modeling submission score: defaults= 0.75119, tuned= 0.77033
#submit_gbc = ensemble.GradientBoostingClassifier()
#submit_gbc = model_selection.GridSearchCV(ensemble.GradientBoostingClassifier(), param_grid={'learning_rate': grid_ratio, 'n_estimators': grid_n_estimator, 'max_depth': grid_max_depth, 'random_state':grid_seed}, scoring = 'roc_auc', cv = cv_split)
#submit_gbc.fit(data1[data1_x_bin], data1[Target])
#print('Best Parameters: ', submit_gbc.best_params_) #Best Parameters:  {'learning_rate': 0.25, 'max_depth': 2, 'n_estimators': 50, 'random_state': 0}
#data_val['Survived'] = submit_gbc.predict(data_val[data1_x_bin])

#extreme boosting w/full dataset modeling submission score: defaults= 0.73684, tuned= 0.77990
#submit_xgb = XGBClassifier()
#submit_xgb = model_selection.GridSearchCV(XGBClassifier(), param_grid= {'learning_rate': grid_learn, 'max_depth': [0,2,4,6,8,10], 'n_estimators': grid_n_estimator, 'seed': grid_seed}, scoring = 'roc_auc', cv = cv_split)
#submit_xgb.fit(data1[data1_x_bin], data1[Target])
#print('Best Parameters: ', submit_xgb.best_params_) #Best Parameters:  {'learning_rate': 0.01, 'max_depth': 4, 'n_estimators': 300, 'seed': 0}
#data_val['Survived'] = submit_xgb.predict(data_val[data1_x_bin])


#hard voting classifier w/full dataset modeling submission score: defaults=-, tuned = 0.74655 V4
# data_val['Transported'] = vote_hard.predict(data_val[data1_x_bin])  # 0.74655 V4
# data_val['Transported'] = grid_hard.predict(data_val[data1_x_bin])  # 0.70189 V4


#soft voting classifier w/full dataset modeling submission score: defaults=-, tuned = 0.75005 V6
# data_val['Transported'] = vote_soft.predict(data_val[data1_x_bin])  # 0.75005 V6
# data_val['Transported'] = grid_soft.predict(data_val[data1_x_bin])  # 0.74982 V5


#submit file
submit = data_val[['PassengerId','Transported']]
submit.to_csv("submission.csv", index=False)

print('Validation Data Distribution: \n', data_val['Transported'].value_counts(normalize = True))
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
#TODO