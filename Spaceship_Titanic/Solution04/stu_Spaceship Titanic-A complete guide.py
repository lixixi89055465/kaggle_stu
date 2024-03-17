# -*- coding: utf-8 -*-
# @Time : 2024/3/17 11:26
# @Author : nanji
# @Site :
# @File : stu_Spaceship Titanic-A complete guide.py
# @Software: PyCharm
# @Comment : https://www.kaggle.com/code/samuelcortinhas/spaceship-titanic-a-complete-guide#Missing-values

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn

print("Sklearn version is {}".format(sklearn.__version__))
import seaborn as sns

sns.set(style='darkgrid', font_scale=1.4)
from imblearn.over_sampling import SMOTE
import itertools
import warnings

warnings.filterwarnings('ignore')
import plotly.express as px
import time

# Sklearn
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score, precision_score, f1_score
from sklearn.metrics import roc_auc_score, plot_confusion_matrix, plot_roc_curve, roc_curve
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, LabelEncoder
from sklearn.feature_selection import mutual_info_classif
from sklearn.decomposition import PCA
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import eli5
from eli5.sklearn import PermutationImportance
from sklearn.utils import resample

# Models
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.naive_bayes import GaussianNB

train = pd.read_csv('../data/train.csv')
test = pd.read_csv('../data/test.csv')

print('Train set shape:', train.shape)
print('Test set shape:', test.shape)
print(train.head())
print('TRAIN SET MISSING VALUES:')
print(train.isna().sum())
print('')
print('TEST SET MISSING VALUES:')
print(test.isna().sum())
# print(f'Duplates in train set:{train.duplicated().sum()},({np.round(100 * train.duplicated().sum() / len(train), 1)}%)')

print(
	f'Duplicated in train set :{train.duplicated().sum()},({np.round(100 * train.duplicated().sum() / len(train), 1)}%)')
print(
	f'Duplicated in test set :{test.duplicated().sum()},({np.round(100 * train.duplicated().sum() / len(train), 1)}%)')
print(train.nunique())
print(train.dtypes)
# plt.figure(figsize=(6, 6,))
# train['Transported'].value_counts().plot.pie(explode=[0.1, 0.1], autopct='%1.1f%%', shadow=True,
#                                              textprops={'fontsize': 16}).set_title('Target distribution')

# plt.figure(figsize=(10, 4))
# Histogram
# sns.histplot(data=train, x='Age', hue='Transported', binwidth=1, kde=True)
# plt.show()

# Expenditure features

# exp_feats = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
# fig = plt.figure(figsize=(10, 20))
# for i, var_name in enumerate(exp_feats):
#     ax = fig.add_subplot(5, 2, 2 * i + 1)
#     sns.histplot(data=train, x=var_name, axes=ax, bins=30, kde=False, hue='Transported')
#     ax.set_title(var_name)
#     ax = fig.add_subplot(5, 2, 2 * i + 2)
#     sns.histplot(data=train, x=var_name, axes=ax, bins=30, kde=True, hue='Transported')
#     plt.ylim([0, 100])
#     ax.set_title(var_name)
#
# fig.tight_layout()
# plt.show()

# Categorical features
# cat_feats = ['HomePlanet', 'CryoSleep', 'Destination', 'VIP']
# fig = plt.figure(figsize=(10, 16))
#
# for i, var_name in enumerate(cat_feats):
#     ax = fig.add_subplot(4, 1, i + 1)
#     sns.countplot(data=train, x=var_name, axes=ax, hue='Transported')
#     ax.set_title(var_name)
#
# fig.tight_layout()
# plt.show()

print('1' * 100)
qual_feats = ['PassengerId', 'Cabin', 'Name']
print(train[qual_feats].head())
# plt.show()
print('2' * 100)
train['Age_group'] = np.nan
train.loc[train['Age'] <= 12, 'Age_group'] = 'Age_0-12'
train.loc[(train['Age'] > 12) & (train['Age'] < 18), 'Age_group'] = 'Age_13-17'
train.loc[(train['Age'] >= 18) & (train['Age'] <= 25), 'Age_group'] = 'Age_18-25'
train.loc[(train['Age'] >= 26) & (train['Age'] <= 30), 'Age_group'] = 'Age_26-30'
train.loc[(train['Age'] > 30) & (train['Age'] <= 50), 'Age_group'] = 'Age_31-50'
train.loc[(train['Age'] > 50), 'Age_group'] = 'Age_51+'

test['Age_group'] = np.nan
test.loc[train['Age'] <= 12, 'Age_group'] = 'Age_0-12'
test.loc[(train['Age'] > 12) & (train['Age'] < 18), 'Age_group'] = 'Age_13-17'
test.loc[(train['Age'] >= 18) & (train['Age'] <= 25), 'Age_group'] = 'Age_18-25'
test.loc[(train['Age'] >= 26) & (train['Age'] <= 30), 'Age_group'] = 'Age_26-30'
test.loc[(train['Age'] > 30) & (train['Age'] <= 50), 'Age_group'] = 'Age_31-50'
test.loc[(train['Age'] > 50), 'Age_group'] = 'Age_51+'
# Plot distribution of new features

# plt.figure(figsize=(10, 4))
# g = sns.countplot(data=train, x='Age_group', hue='Transported',
#                   order=['Age_0-12', 'Age_13-17', 'Age_18-25', 'Age_26-30', 'Age_31-50', 'Age_51+'])
# plt.title('Age group distribution')
# plt.show()

# New features - training set
# Expenditure features
exp_feats = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
train['Expenditure'] = train[exp_feats].sum(axis=1)
train['No_spending'] = (train['Expenditure'] == 0).astype(int)
# New features  - test set
test['Expenditure'] = test[exp_feats].sum(axis=1)
test['No_spending'] = (test['Expenditure']).astype(int)
# Plot distribution of new features

# fig = plt.figure(figsize=(12, 4))
# plt.subplot(1, 2, 1)
# sns.histplot(data=train, x='Expenditure', hue='Transported', bins=200)
#
# plt.title('Total expenditure (truncated) ')
# plt.ylim([0, 200])
# plt.xlim([0, 20000])
#
# plt.subplot(1, 2, 2)
# sns.countplot(data=train, x='No_spending', hue='Transported')
# plt.title('No spending indicator')
# fig.tight_layout()
# plt.show()
train['Group'] = train['PassengerId'].apply(lambda x: x.split('_')[0]).astype(int)
test['Group'] = test['PassengerId'].apply(lambda x: x.split('_')[0]).astype(int)

# New feature - group size
train['Group_size'] = train['Group'].map(lambda x: pd.concat([train['Group'], test['Group']]).value_counts()[x])
test['Group_size'] = test['Group'].map(lambda x: pd.concat([train['Group'], test['Group']]).value_counts()[x])
a = pd.concat([train['Group'], test['Group']])
print('8' * 100)
print(a.head())

# Plot distribution of new features
# fig=plt.figure(figsize=(12,4))
# plt.figure(figsize=(20,4))
# plt.subplot(1,2,1)
# sns.histplot(data=train,x='Group',hue='Transported',binwidth=1)
# plt.title('Group')
#
# plt.subplot(1,2,2)
# sns.countplot(data=train, x='Group_size', hue='Transported')
# plt.title('Group size')
# fig.tight_layout()


# Expenditure features
exp_feats = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']

# Plot expenditure features
# fig = plt.figure(figsize=(10, 20))
# for i, var_name in enumerate(exp_feats):
# 	# Left plot
# 	ax = fig.add_subplot(5, 2, 2 * i + 1)
# 	sns.histplot(data=train, x=var_name, axes=ax, bins=30, kde=False, hue='Transported')
# 	ax.set_title(var_name)
#
# 	# Right plot (truncated)
# 	ax = fig.add_subplot(5, 2, 2 * i + 2)
# 	sns.histplot(data=train, x=var_name, axes=ax, bins=30, kde=True, hue='Transported')
# 	plt.ylim([0, 100])
# 	ax.set_title(var_name)
# fig.tight_layout()  # Improves appearance a bit
# plt.show()

# Categorical features
cat_feats = ['HomePlanet', 'CryoSleep', 'Destination', 'VIP']

# Plot categorical features
# fig = plt.figure(figsize=(10, 16))
# for i, var_name in enumerate(cat_feats):
# 	ax = fig.add_subplot(4, 1, i + 1)
# 	sns.countplot(data=train, x=var_name, axes=ax, hue='Transported')
# 	ax.set_title(var_name)
# fig.tight_layout()  # Improves appearance a bit
# plt.show()
# Qualitative features
qual_feats = ['PassengerId', 'Cabin', 'Name']

# Preview qualitative features
print('0' * 100)
print(train[qual_feats].head())

# New features - training set
train['Age_group'] = np.nan
train.loc[train['Age'] <= 12, 'Age_group'] = 'Age_0-12'
train.loc[(train['Age'] > 12) & (train['Age'] < 18), 'Age_group'] = 'Age_13-17'
train.loc[(train['Age'] >= 18) & (train['Age'] <= 25), 'Age_group'] = 'Age_18-25'
train.loc[(train['Age'] > 25) & (train['Age'] <= 30), 'Age_group'] = 'Age_26-30'
train.loc[(train['Age'] > 30) & (train['Age'] <= 50), 'Age_group'] = 'Age_31-50'
train.loc[train['Age'] > 50, 'Age_group'] = 'Age_51+'
# New features - test set
test['Age_group'] = np.nan
test.loc[test['Age'] <= 12, 'Age_group'] = 'Age_0-12'
test.loc[(test['Age'] > 12) & (test['Age'] < 18), 'Age_group'] = 'Age_13-17'
test.loc[(test['Age'] >= 18) & (test['Age'] <= 25), 'Age_group'] = 'Age_18-25'
test.loc[(test['Age'] > 25) & (test['Age'] <= 30), 'Age_group'] = 'Age_26-30'
test.loc[(test['Age'] > 30) & (test['Age'] <= 50), 'Age_group'] = 'Age_31-50'
test.loc[test['Age'] > 50, 'Age_group'] = 'Age_51+'

# New features - training set
train['Expenditure'] = train[exp_feats].sum(axis=1)
train['No_spending'] = (train['Expenditure'] == 0).astype(int)
# New features - test set
test['Expenditure'] = test[exp_feats].sum(axis=1)
test['No_spending'] = (test['Expenditure'] == 0).astype(int)

# # Plot distribution of new features
# fig=plt.figure(figsize=(12,4))
# plt.subplot(1,2,1)
# sns.histplot(data=train, x='Expenditure', hue='Transported', bins=200)
# plt.title('Total expenditure (truncated)')
# plt.ylim([0,200])
# plt.xlim([0,20000])
#
# plt.subplot(1,2,2)
# sns.countplot(data=train, x='No_spending', hue='Transported')
# plt.title('No spending indicator')
# fig.tight_layout()

# New feature - Group
train['Group'] = train['PassengerId'].apply(lambda x: x.split('_')[0]).astype(int)
test['Group'] = test['PassengerId'].apply(lambda x: x.split('_')[0]).astype(int)

# New feature - Group size
train['Group_size'] = train['Group'].map(lambda x: pd.concat([train['Group'], test['Group']]).value_counts()[x])
test['Group_size'] = test['Group'].map(lambda x: pd.concat([train['Group'], test['Group']]).value_counts()[x])

# Plot distribution of new features
# plt.figure(figsize=(20,4))
# plt.subplot(1,2,1)
# sns.histplot(data=train, x='Group', hue='Transported', binwidth=1)
# plt.title('Group')
#
# plt.subplot(1,2,2)
# sns.countplot(data=train, x='Group_size', hue='Transported')
# plt.title('Group size')
# fig.tight_layout()

# New feature
train['Solo'] = (train['Group_size'] == 1).astype(int)
test['Solo'] = (test['Group_size'] == 1).astype(int)

# # New feature distribution
# plt.figure(figsize=(10,4))
# sns.countplot(data=train, x='Solo', hue='Transported')
# plt.title('Passenger travelling solo or not')
# plt.ylim([0,3000])


# Replace NaN's with outliers for now (so we can split feature)
train['Cabin'].fillna('Z/9999/Z', inplace=True)
test['Cabin'].fillna('Z/9999/Z', inplace=True)

# New features - training set
train['Cabin_deck'] = train['Cabin'].apply(lambda x: x.split('/')[0])
train['Cabin_number'] = train['Cabin'].apply(lambda x: x.split('/')[1]).astype(int)
train['Cabin_side'] = train['Cabin'].apply(lambda x: x.split('/')[2])

# New features - test set
test['Cabin_deck'] = test['Cabin'].apply(lambda x: x.split('/')[0])
test['Cabin_number'] = test['Cabin'].apply(lambda x: x.split('/')[1]).astype(int)
test['Cabin_side'] = test['Cabin'].apply(lambda x: x.split('/')[2])

# Put Nan's back in (we will fill these later)
train.loc[train['Cabin_deck'] == 'Z', 'Cabin_deck'] = np.nan
train.loc[train['Cabin_number'] == 9999, 'Cabin_number'] = np.nan
train.loc[train['Cabin_side'] == 'Z', 'Cabin_side'] = np.nan
test.loc[test['Cabin_deck'] == 'Z', 'Cabin_deck'] = np.nan
test.loc[test['Cabin_number'] == 9999, 'Cabin_number'] = np.nan
test.loc[test['Cabin_side'] == 'Z', 'Cabin_side'] = np.nan

# Drop Cabin (we don't need it anymore)
train.drop('Cabin', axis=1, inplace=True)
test.drop('Cabin', axis=1, inplace=True)

# # Plot distribution of new features
# fig=plt.figure(figsize=(10,12))
# plt.subplot(3,1,1)
# sns.countplot(data=train, x='Cabin_deck', hue='Transported', order=['A','B','C','D','E','F','G','T'])
# plt.title('Cabin deck')
#
#
# plt.subplot(3,1,2)
# sns.histplot(data=train, x='Cabin_number', hue='Transported',binwidth=20)
# plt.vlines(300, ymin=0, ymax=200, color='black')
# plt.vlines(600, ymin=0, ymax=200, color='black')
# plt.vlines(900, ymin=0, ymax=200, color='black')
# plt.vlines(1200, ymin=0, ymax=200, color='black')
# plt.vlines(1500, ymin=0, ymax=200, color='black')
# plt.vlines(1800, ymin=0, ymax=200, color='black')
# plt.title('Cabin number')
# plt.xlim([0,2000])
#
# plt.subplot(3,1,3)
# sns.countplot(data=train, x='Cabin_side', hue='Transported')
# plt.title('Cabin side')
# fig.tight_layout()
'''
This is interesting! It appears that Cabin_number is grouped into chunks of 300 cabins. This means we can compress this feature into a categorical one, which indicates which chunk each passenger is in.

Other notes: The cabin deck 'T' seems to be an outlier (there are only 5 samples).
'''

# New features - training set
train['Cabin_region1'] = (train['Cabin_number'] < 300).astype(int)  # one-hot encoding
train['Cabin_region2'] = ((train['Cabin_number'] >= 300) & (train['Cabin_number'] < 600)).astype(int)
train['Cabin_region3'] = ((train['Cabin_number'] >= 600) & (train['Cabin_number'] < 900)).astype(int)
train['Cabin_region4'] = ((train['Cabin_number'] >= 900) & (train['Cabin_number'] < 1200)).astype(int)
train['Cabin_region5'] = ((train['Cabin_number'] >= 1200) & (train['Cabin_number'] < 1500)).astype(int)
train['Cabin_region6'] = ((train['Cabin_number'] >= 1500) & (train['Cabin_number'] < 1800)).astype(int)
train['Cabin_region7'] = (train['Cabin_number'] >= 1800).astype(int)

# New features - test set
test['Cabin_region1'] = (test['Cabin_number'] < 300).astype(int)  # one-hot encoding
test['Cabin_region2'] = ((test['Cabin_number'] >= 300) & (test['Cabin_number'] < 600)).astype(int)
test['Cabin_region3'] = ((test['Cabin_number'] >= 600) & (test['Cabin_number'] < 900)).astype(int)
test['Cabin_region4'] = ((test['Cabin_number'] >= 900) & (test['Cabin_number'] < 1200)).astype(int)
test['Cabin_region5'] = ((test['Cabin_number'] >= 1200) & (test['Cabin_number'] < 1500)).astype(int)
test['Cabin_region6'] = ((test['Cabin_number'] >= 1500) & (test['Cabin_number'] < 1800)).astype(int)
test['Cabin_region7'] = (test['Cabin_number'] >= 1800).astype(int)

# # Plot distribution of new features
# plt.figure(figsize=(10,4))
# train['Cabin_regions_plot']=(train['Cabin_region1']+2*train['Cabin_region2']+3*train['Cabin_region3']+4*train['Cabin_region4']+5*train['Cabin_region5']+6*train['Cabin_region6']+7*train['Cabin_region7']).astype(int)
# sns.countplot(data=train, x='Cabin_regions_plot', hue='Transported')
# plt.title('Cabin regions')
# train.drop('Cabin_regions_plot', axis=1, inplace=True)

# Replace NaN's with outliers for now (so we can split feature)
train['Name'].fillna('Unknown Unknown', inplace=True)
test['Name'].fillna('Unknown Unknown', inplace=True)

# New feature - Surname
train['Surname'] = train['Name'].str.split().str[-1]
test['Surname'] = test['Name'].str.split().str[-1]

# New feature - Family size
train['Family_size'] = train['Surname'].map(lambda x: pd.concat([train['Surname'], test['Surname']]).value_counts()[x])
test['Family_size'] = test['Surname'].map(lambda x: pd.concat([train['Surname'], test['Surname']]).value_counts()[x])
# Put Nan's back in (we will fill these later)
train.loc[train['Surname']=='Unknown','Surname']=np.nan
train.loc[train['Family_size']>100,'Family_size']=np.nan
test.loc[test['Surname']=='Unknown','Surname']=np.nan
test.loc[test['Family_size']>100,'Family_size']=np.nan

# Drop name (we don't need it anymore)
train.drop('Name', axis=1, inplace=True)
test.drop('Name', axis=1, inplace=True)

# # New feature distribution
# plt.figure(figsize=(12,4))
# sns.countplot(data=train, x='Family_size', hue='Transported')
# plt.title('Family size')

'''
Missing values¶
Combine train and test

This will make it easier to fill missing values. We will split it back later.
'''
# Labels and features
y=train['Transported'].copy().astype(int)
X=train.drop('Transported', axis=1).copy()

# Concatenate dataframes
data=pd.concat([X, test], axis=0).reset_index(drop=True)

# Columns with missing values
na_cols=data.columns[data.isna().any()].tolist()

# Missing values summary
mv=pd.DataFrame(data[na_cols].isna().sum(), columns=['Number_missing'])
mv['Percentage_missing']=np.round(100*mv['Number_missing']/len(data),2)
# # mv
# # Heatmap of missing values
# plt.figure(figsize=(12,6))
# sns.heatmap(train[na_cols].isna().T, cmap='summer')
# plt.title('Heatmap of missing values')

# Countplot of number of missing values by passenger
train['na_count']=train.isna().sum(axis=1)
plt.figure(figsize=(10,4))
sns.countplot(data=train, x='na_count', hue='Transported')
plt.title('Number of missing entries by passenger')
train.drop('na_count', axis=1, inplace=True)

'''

Notes:

Missing values are independent of the target and for the most part are isolated.
Even though only 2% of the data is missing, about 25% of all passengers have at least 1 missing value.
PassengerId is the only (original) feature to not have any missing values.
Insight:

Since most of the missing values are isolated it makes sense to try to fill these in as opposed to just dropping rows.
If there is a relationship between PassengerId and other features we can fill missing values according to this column.
'''

'''
Strategy

The easiest way to deal with missing values is to just use the median for continuous features and 
the mode for categorical features (see version 20 of this notebook). 
This will work 'well enough' but if we want to maximise the accuracy of our models then we need to look for patterns within the missing data. The way to do this is by looking at the joint distribution of features, e.g. do passengers from the same group tend to come from the same family? There are obviously many combinations so we will just summarise the useful trends I and others have found.

HomePlanet and Group
'''

# Joint distribution of Group and HomePlanet
GHP_gb=data.groupby(['Group','HomePlanet'])['HomePlanet'].size().unstack().fillna(0)
print('1'*100)
print(GHP_gb.head())


