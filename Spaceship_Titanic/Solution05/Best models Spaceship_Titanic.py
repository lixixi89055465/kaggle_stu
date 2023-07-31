'''
Hello and Welcome to Kaggle, the online Data Science Community to learn, share, and compete. Most beginners get lost in the field, because they fall into the black box approach, using libraries and algorithms they don't understand. This tutorial will give you a 1-2-year head start over your peers, by providing a framework that teaches you how-to think like a data scientist vs what to think/code. Not only will you be able to submit your first competition, but you’ll be able to solve any problem thrown your way. I provide clear explanations, clean code, and plenty of links to resources. Please Note: This Kernel is still being improved. So check the Change Logs below for updates. Also, please be sure to upvote, fork, and comment and I'll continue to develop. Thanks, and may you have "statistically significant" luck!
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
from sklearn import metrics
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

import random
import time

import warnings

warnings.filterwarnings('ignore')
print('-' * 25)
from subprocess import check_output

print(check_output(['ls', '../data/']))

import pandas as pd

data_raw = pd.read_csv('../data/train.csv')
data_val = pd.read_csv('../data/test.csv')
# #to play with our data we'll create a copy
# #remember python assignment or equal passes by reference vs values, so we use the copy function: https://stackoverflow.com/questions/46327494/python-pandas-dataframe-copydeep-false-vs-copydeep-true-vs
data1 = data_raw.copy(deep=True)
data_cleaner = [data1, data_val]
# print("2" * 100)
# print(data_raw.info())
# print(data_raw.head())
# print(data_raw.sample(9))
# Duplicates
import numpy as np

# print("3" * 100)
# print(
#     f'Duplicates in train set :{data_raw.duplicated().sum()},'
#     f'{np.round(100 * data_raw.duplicated().sum() / len(data_raw), 1)})')
# print(
#     f'Duplicates in test set :{data_raw.duplicated().sum()},'
#     f'{np.round(100 * data_val.duplicated().sum() / len(data_raw), 1)})')
#
# print('Train columns with null value:\n', data1.isnull().sum())
# print("-" * 100)
# print('Test/Validation columns with null value :\n', data_val.isnull().sum())
# print("- " * 100)
# print(data_raw.nunique())
# print(data_raw.dtypes)
# Expenditure features
exp_feats = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
# Categorical feature
cat_feats = ['HomePlanet', 'CryoSleep', 'Destination', 'VIP']
# Qualitative features
qual_feats = ['PassengerId', 'Cabin', 'Name']

# for dataset in data_cleaner:
#     dataset['Age_group'] = np.nan
#     dataset.loc[dataset['Age'] <= 12, 'Age_group'] = 'Age_0-12'
#     dataset.loc[(dataset['Age'] > 12) & (dataset['Age'] < 18), 'Age_group'] = 'Age_13-17'
#     dataset.loc[(dataset['Age'] >= 18) & (dataset['Age'] <= 25), 'Age_group'] = 'Age_18-25'
#     dataset.loc[(dataset['Age'] > 25) & (dataset['Age'] <= 30), 'Age_group'] = 'Age_26-30'
#     dataset.loc[(dataset['Age'] > 30) & (dataset['Age'] <= 50), 'Age_group'] = 'Age_31-50'
#     dataset.loc[dataset['Age'] > 50, 'Age_group'] = 'Age_51+'

# Plot distribution of new features
# plt.figure(figsize=(10, 4))
# g = sns.countplot(data=data1, x='Age_group', hue='Transported',
#                   order=['Age_0-12', 'Age_13-17', 'Age_18-25', 'Age_26-30', 'Age_31-50', 'Age_51+'])
# plt.title('Age group distribution')
# plt.show()

# for dataset in data_cleaner:
#     dataset['Expenditure'] = dataset[exp_feats].sum(axis=1)
#     dataset['No_spending'] = (dataset['Expenditure'] == 0).astype(int)
# fig = plt.figure(figsize=(12, 4))
# plt.subplot(1, 2, 1)
# sns.histplot(data=data1, x='Expenditure', hue='Transported', bins=200)
# plt.title('Total expenditure (truncated) ')
# plt.ylim([0, 200])
# plt.xlim([0, 20000])
# plt.subplot(1, 2, 2)
# sns.countplot(data=data1, x='No_spending', hue='Transported')
# plt.title('No spending indicator')
# fig.tight_layout()
# for dataset in data_cleaner:
#     dataset['Expenditure'] = dataset[exp_feats].sum(axis=1)
#     dataset['No_spending'] = (dataset['Expenditure'] == 0).astype(int)

# Plot distribution of new features
# fig = plt.figure(figsize=(12, 4))
# plt.subplot(1, 2, 1)
# sns.histplot(data=data1, x='Expenditure', hue='Transported', bins=200)
# plt.title('Total expenditure (truncated)')
# plt.ylim([0, 200])
# plt.xlim([0, 20000])
#
# plt.subplot(1, 2, 2)
# sns.countplot(data=data1, x='No_spending', hue='Transported')
# plt.title('No spending indicator')
# fig.tight_layout()
for dataset in data_cleaner:
    dataset['Group'] = dataset['PassengerId'].apply(lambda x: x.split('_')[0]).astype(int)
    dataset['Group_size'] = dataset['Group'].map(lambda x: dataset['Group'].value_counts()[x])

# plt.figure(figsize=(20, 4))
# plt.subplot(1, 2, 1)
# sns.histplot(data=data1, x='Group', hue='Transported', binwidth=1)
# plt.title('Group')
#
# plt.subplot(1,2,2)
# sns.countplot(data=data1,x='Group_size',hue='Transported')
# plt.title('Group size')
# fig.tight_layout()
# plt.show()

print("0" * 100)
# print(dataset['Group'].value_counts())
## New features
# data_raw['Solo'] = (data_raw['Group_size'] == 1).astype(int)
# data_val['Solo'] = (data_val['Group_size'] == 1).astype(int)

print('1' * 100)
for dataset in data_cleaner:
    print(dataset.columns)
    dataset['Solo'] = (dataset['Group_size'] == 1).astype(int)
# New feature distribution
# plt.figure(figsize=(10,4))
# sns.countplot(data=data1,x='Solo',hue='Transported')
# plt.title('Passenger travelling sole  or not ')
# plt.ylim([0,3000])
# plt.show()

for dataset in data_cleaner:
    dataset['Cabin'].fillna('Z/9999/Z', inplace=True)
    dataset[['Cabin_deck', 'Cabin_number', 'Cabin_side']] = dataset['Cabin'].str.split('/', expand=True)
    dataset['Cabin_number'] = dataset['Cabin_number'].astype(int)
    dataset.loc[dataset['Cabin_deck'] == 'Z', 'Cabin_deck'] = np.nan
    dataset.loc[dataset['Cabin_number'] == '9999', 'Cabin_number'] = np.nan
    dataset.loc[dataset['Cabin_side'] == 'Z', 'Cabin_side'] = np.nan
    dataset.drop('Cabin', axis=1, inplace=True)
# plot distribution of new features
# fig = plt.figure(figsize=(20, 4))
# plt.subplot(3, 1, 1)
# sns.countplot(data=data1, x='Cabin_deck', hue='Transported', order=['A', 'B', 'C', 'D', 'E', 'F', 'G', 'T'])
# plt.title('Cabin-deck')

# plt.subplot(3, 1, 2)
# sns.histplot(data=data1, x='Cabin_number', hue='Transported', binwidth=20)
# plt.vlines(300, ymin=0, ymax=200, color='black')
# plt.vlines(600, ymin=0, ymax=200, color='black')
# plt.vlines(900, ymin=0, ymax=200, color='black')
# plt.vlines(1200, ymin=0, ymax=200, color='black')
# plt.vlines(1500, ymin=0, ymax=200, color='black')
# plt.vlines(1800, ymin=0, ymax=200, color='black')
# plt.title('Cabin number')
# plt.xlim([0, 2000])

# plt.subplot(3, 1, 3)
# sns.countplot(data=data1, x='Cabin_side', hue='Transported')
# plt.title('cabin side')
# fig.tight_layout()
# print("5" * 100)
# for dataset in data_cleaner:
#     # New feature - training set
#     dataset['Cabin_region1'] = (dataset['Cabin_number'] < 300).astype(int)
#     dataset['Cabin_region2'] = ((dataset['Cabin_number'] >= 300) & (dataset['Cabin_number'] < 600)).astype(int)
#     dataset['Cabin_region3'] = ((dataset['Cabin_number'] >= 600) & (dataset['Cabin_number'] < 900)).astype(int)
#     dataset['Cabin_region4'] = ((dataset['Cabin_number'] >= 900) & (dataset['Cabin_number'] < 1200)).astype(int)
#     dataset['Cabin_region5'] = ((dataset['Cabin_number'] >= 1200) & (dataset['Cabin_number'] < 1500)).astype(int)
#     dataset['Cabin_region6'] = ((dataset['Cabin_number'] >= 1500) & (dataset['Cabin_number'] < 1500)).astype(int)
#     dataset['Cabin_region7'] = (dataset['Cabin_number'] >= 1800).astype(int)

# plt.figure(figsize=(10, 4))
# data1['Cabin_region_plot'] = (
#         data1['Cabin_region1'] + 2 * data1['Cabin_region2'] + 3 * data1['Cabin_region3'] + 4 * data1[
#     'Cabin_region4'] + 5 * data1['Cabin_region5'] + 6 * data1['Cabin_region6'] + 7 * data1['Cabin_region7']).astype(
#     int)
# sns.countplot(data=data1, x='Cabin_region_plot', hue='Transported')
# plt.title('Cabin region ')
# data1.drop('Cabin_region_plot', axis=1, inplace=True)
#
for dataset in data_cleaner:
    dataset['Name'].fillna('Unknown Unknown', inplace=True)
    dataset['Surname'] = dataset['Name'].str.split().str[-1]
    dataset['Family_size'] = dataset['Surname'].map(lambda x: dataset['Surname'].value_counts()[x])
    dataset.loc[dataset['Surname'] == 'Unknow Unknow'] = np.nan
    dataset.loc[dataset['Family_size'] > 100, 'Family_size'] = np.nan
    dataset.drop('Name', axis=1, inplace=True)

# New  feature distribution
#
# plt.figure(figsize=(12, 4))
# sns.countplot(data=data1, x='Family_size', hue='Transported')
# plt.title('Family_size ')
# Missing values¶

# data1['Transported'].astype(int)
# for dataset in data_cleaner:
#     # Columns with missing values
#     na_cols = dataset.columns[dataset.isna().any()].tolist()
#     mv = pd.DataFrame(dataset[na_cols].isna().sum(), columns=['Number_missing'])
#     mv['Percentage_missing'] = np.round(100 * mv['Number_missing'] / len(dataset), 2)
# print(mv, '\n')

# Countplot of number of missing values by passenger
# data1['na_count'] = data1.isna().sum(axis=1)
# plt.figure(figsize=(10, 4))
# sns.countplot(data=data1, x='na_count', hue='Transported')
# plt.title("number of missing entries by pasenger")
# data1.drop('na_count', axis=1, inplace=True)

# We managed to fill 131 values with 100% confidence but we are nott finished yet.


for dataset in data_cleaner:
    GHP_gb = dataset.groupby(['Group', 'HomePlanet'])['HomePlanet'].size().unstack().fillna(0)
    # Missing values before
    HP_bef = dataset['HomePlanet'].isna().sum()
    GHP_index = dataset[dataset['HomePlanet'].isna()][
        (dataset[dataset['HomePlanet'].isna()]['Group']).isin(GHP_gb.index)].index
    dataset.loc[GHP_index, 'HomePlanet'] = dataset.iloc[GHP_index, :]['Group'].map(
        lambda x: GHP_gb.idxmax(axis=1)[x])
    # print("#Missing values before:", HP_bef)
    # print("#Missing values afeter:", dataset['HomePlanet'].isna().sum())

print('5' * 100)
# for dataset in data_cleaner:
#     HP_bef = dataset['HomePlanet'].isna().sum()
#     dataset.loc[
#         (dataset['HomePlanet'].isna()) & (dataset['Cabin_deck'].isin(['A', 'B', 'C', 'T'])), 'HomePlanet'] = 'Europe'
#     dataset.loc[
#         (dataset['HomePlanet'].isna()) & (dataset['Cabin_deck'] == 'G'), 'HomePlanet'] = 'Earth'
# print('#HomePlanet missing values before:', HP_bef)
# print('#HomePlanet missing values after:', dataset['HomePlanet'].isna().sum())

print('6' * 100)
for data in data_cleaner:
    SHP_gb = data.groupby(['Surname', 'HomePlanet']).size().unstack().fillna(0)
    HP_bef = data['HomePlanet'].isna().sum()
    SHP_index = data[data['HomePlanet'].isna()][(data[data['HomePlanet'].isna()]['Surname']).isin(SHP_gb.index)].index
    data.loc[SHP_index, 'HomePlanet'] = data.iloc[SHP_index, :]['Surname'].map(lambda x: SHP_gb.idxmax(axis=1)[x])
    # Print number of missing values left
    # print('#HomePlanet missing values before:', HP_bef)
    # print('#HomePlanet missing values before:', data['HomePlanet'].isna().sum())

print('7' * 100)
# for data in data_cleaner:
#     HP_bef = data['HomePlanet'].isna().sum()
#     data.loc[(data['HomePlanet'].isna()) & ~(data['Cabin_deck'] == 'D'), 'HomePlanet'] = 'Earth'
#     data.loc[(data['HomePlanet'].isna()) & (data['Cabin_deck'] == 'D'), 'HomePlanet'] = 'Mars'
# print('#HomePlanet missing values before:', HP_bef)
# print('#HomePlanet missing values after:', data['HomePlanet'].isna().sum())
# We're done with HomePlanet.
print('8' * 100)
# for data in data_cleaner:
#     D_bef = data['Destination'].isna().sum()
#     data.loc[data['Destination'].isna(), 'Destination'] = 'TRAPPIST-1e'
print('9' * 100)
# for data in data_cleaner:
#     GSN_gb = data[data['Group_size'] > 1].groupby(['Group', 'Surname'])['Surname'].size().unstack().fillna(0)
#     SN_bef = data['Surname'].isna().sum()
#     GSN_index = data[data['Surname'].isna()][(data[data['Surname'].isna()]['Group']).isin(GSN_gb.index)].index
#     # Fill corresponding missing values
#     data.loc[GSN_index, 'Surname'] = data.iloc[GSN_index, :]['Group'].map(lambda x: GSN_gb.idxmax(axis=1)[x])
#     # Print number of missing values left
#     # print('#Surname missing values before:', SN_bef)
#     # print('#Surname missing values after:', data['Surname'].isna().sum())
#     data['Surname'].fillna('Unknown', inplace=True)
#     data['Family_size'] = data['Surname'].map(lambda x: data['Surname'].value_counts()[x])
#     data.loc[data['Surname'] == 'Unknow', 'Surname'] = np.nan
#     data.loc[data['Family_size'] > 100, 'Family_size'] = 0

print('0' * 100)
for data in data_cleaner:
    GCD_gb = data[data['Group_size'] > 1].groupby(['Group', 'Cabin_deck'])['Cabin_deck'].size().unstack().fillna(0)
    GCN_gb = data[data['Group_size'] > 1].groupby(['Group', 'Cabin_number'])['Cabin_number'].size().unstack().fillna(0)
    GCS_gb = data[data['Group_size'] > 1].groupby(['Group', 'Cabin_side'])['Cabin_side'].size().unstack().fillna(0)
    CS_bef = data['Cabin_side'].isna().sum()
    GCS_index = data[data['Cabin_side'].isna()][(data[data['Cabin_side'].isna()]['Group']).isin(GCS_gb.index)].index
    data.loc[GCS_index, 'Cabin_side'] = data.iloc[GCS_index, :]['Group'].map(lambda x: GCS_gb.idxmax(axis=1)[x])
    # Print number of missing values left
    print('#Cabin_side missing values before:', CS_bef)
    print('#Cabin_side missing values after:', data['Cabin_side'].isna().sum())
print('1'*100)
for data in data_cleaner:
    # Joint distribution of Surname and Cabin side
    SCS_gb=data[data['Group_size']>1].groupby(['Surname','Cabin_side'])['Cabin_side'].size().unstack().fillna(0)

    # Ratio of sides
    # SCS_gb['Ratio']=SCS_gb[data]/(SCS_gb['P']+SCS_gb['S'])
    print(SCS_gb['P'])

    # Histogram of ratio
    plt.figure(figsize=(10,4))
    sns.histplot(SCS_gb['Ratio'], kde=True, binwidth=0.05)
    plt.title('Ratio of cabin side by surname')