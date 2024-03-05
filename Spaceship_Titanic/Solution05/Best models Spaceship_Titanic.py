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

# print(check_output(['ls', '../data/']).decode('utf8'))

import pandas as pd

data_raw = pd.read_csv('../data/train.csv')
data_val = pd.read_csv('../data/test.csv')
# #to play with our data we'll create a copy
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
# g = sns.countplot(data=data1, x='Age_group', hue='Transported',
#                   order=['Age_0-12', 'Age_13-17', 'Age_18-25', 'Age_26-30', 'Age_31-50', 'Age_51+'])
# plt.title('Age group distribution')
# plt.show()

for dataset in data_cleaner:
    dataset['Expenditure'] = dataset[exp_feats].sum(axis=1)
    dataset['No_spending'] = (dataset['Expenditure'] == 0).astype(int)

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
# plt.show()
# print('0'*100)

# Plot distribution of new features
fig = plt.figure(figsize=(12, 4))
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
    dvc = dataset['Group'].value_counts()
    dataset['Group_size'] = dataset['Group'].map(lambda x: dvc[x])
    # dataset['Solo'] = (dataset['Group_size'] == 1).astype(int)

# plt.figure(figsize=(20, 16))
# plt.subplot(1, 2, 1)
# sns.histplot(data=data1, x='Group', hue='Transported', binwidth=1)
# plt.title('Group')
#
# plt.subplot(1, 2, 2)
# sns.countplot(data=data1, x='Group_size', hue='Transported')
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
# sns.countplot(data=data1, x='Solo', hue='Transported')
# plt.title('Passenger travelling sole  or not ')
# plt.ylim([0, 3000])
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
# fig = plt.figure(figsize=(10, 12))
# plt.subplot(3, 1, 1)
# sns.countplot(data=data1, x='Cabin_deck', hue='Transported', order=['A', 'B', 'C', 'D', 'E', 'F', 'G', 'T'])
# plt.title('Cabin-deck')
#
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
#
# plt.subplot(3, 1, 3)
# sns.countplot(data=data1, x='Cabin_side', hue='Transported')
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
# sns.countplot(data=data1, x='Cabin_region_plot', hue='Transported')
# plt.title('Cabin region ')
# plt.show()
# data1.drop('Cabin_region_plot', axis=1, inplace=True)
#
for dataset in data_cleaner:
    dataset['Name'].fillna('Unknown Unknown', inplace=True)
    dataset['Surname'] = dataset['Name'].str.split().str[-1]
    tset01 = dataset['Name'].str
    dataset['Family_size'] = dataset['Surname'].map(lambda x: dataset['Surname'].value_counts()[x])
    dataset['Surname'].map(lambda x: dataset['Surname'].value_counts()[x])
    # dataset.loc[dataset['Surname'] == 'Unknow Unknow'] = np.nan
    dataset.loc[dataset['Surname'] == 'Unknown', 'Surname'] = np.nan
    dataset.loc[dataset['Family_size'] > 100, 'Family_size'] = np.nan
    dataset.drop('Name', axis=1, inplace=True)

# New  feature distribution
#
# plt.figure(figsize=(12, 4))
# sns.countplot(data=data1, x='Family_size', hue='Transported')
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
# sns.countplot(data=data1, x='na_count', hue='Transported')
# plt.title("number of missing entries by pasenger")
# plt.show()
# data1.drop('na_count', axis=1, inplace=True)

# We managed to fill 131 values with 100% confidence but we are nott finished yet.


#TODO
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
        (dataset['HomePlanet'].isna()) & (dataset['Cabin_deck'].isin(['A', 'B', 'C', 'T'])), 'HomePlanet'] = 'Europe'
    dataset.loc[(dataset['HomePlanet'].isna()) & (dataset['Cabin_deck'] == 'G'), 'HomePlanet'] = 'Earth'
# g1 = \
# dataset[(~dataset['HomePlanet'].isna()) & (~dataset['Cabin_deck'].isna())].groupby(['Cabin_deck', 'HomePlanet'])[
#     'HomePlanet'].size().unstack().fillna(0)
# print('#HomePlanet missing values before:', HP_bef)
# print('#HomePlanet missing values after:', dataset['HomePlanet'].isna().sum())

print('6' * 100)
for data in data_cleaner:
    SHP_gb = data.groupby(['Surname', 'HomePlanet']).size().unstack().fillna(0)
    HP_bef = data['HomePlanet'].isna().sum()
    SHP_index = data[data['HomePlanet'].isna()][(data[data['HomePlanet'].isna()]['Surname']).isin(SHP_gb.index)].index
    SHP_gb_idxmax = SHP_gb.idxmax(axis=1)
    data.loc[SHP_index, 'HomePlanet'] = data.iloc[SHP_index, :]['Surname'].map(lambda x: SHP_gb_idxmax[x])

# Print number of missing values left
# print('#HomePlanet missing values before:', HP_bef)
# print('#HomePlanet missing values after:', data['HomePlanet'].isna().sum())

print('7' * 100)
#TODO
for data in data_cleaner:
    HP_bef = data['HomePlanet'].isna().sum()
    # g1 = data[(~data['HomePlanet'].isna())&(~data['Cabin_deck'].isin(['A','B','C','T','G']))].groupby(['Cabin_deck', 'HomePlanet'])[
    #     'Cabin_deck'].size().unstack().fillna(0)
    data.loc[(data['HomePlanet'].isna()) & ~(data['Cabin_deck'] == 'D'), 'HomePlanet'] = 'Earth'
    data.loc[(data['HomePlanet'].isna()) & (data['Cabin_deck'] == 'D'), 'HomePlanet'] = 'Mars'
# print('#HomePlanet missing values before:', HP_bef)
# print('#HomePlanet missing values after:', data['HomePlanet'].isna().sum())
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
    # print('#Surname missing values after:', data['Surname'].isna().sum())
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
    # data[data['Group_size'] > 1].groupby(['Group', 'Cabin_side'])['Group'].size().unstack().fillna(0)
    # Missing values before
    CS_bef = data['Cabin_side'].isna().sum()
    GCS_index = data[data['Cabin_side'].isna()][(data[data['Cabin_side'].isna()]['Group']).isin(GCS_gb.index)].index
    data.loc[GCS_index, 'Cabin_side'] = data.iloc[GCS_index, :]['Group'].map(lambda x: GCS_gb.idxmax(axis=1)[x])
    # Print number of missing values left
    # print('#Cabin_side missing values before:', CS_bef)
    # print('#Cabin_side missing values after:', data['Cabin_side'].isna().sum())
print('1' * 100)
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
print("3" * 100)
for data in data_cleaner:
    # Value counts
    print(data['Cabin_side'].value_counts())
    # Missing values before
    CS_bef = data['Cabin_side'].isna().sum()
    data.loc[data['Cabin_side'].isna(), 'Cabin_deck'] = 'Z'
    # print('#Cabin side missing values before:', CS_bef)
    # print('#Cabin side missing value after:', data['Cabin_side'].isna().sum())

print("4" * 100)
for data in data_cleaner:
    # print(data['Cabin_deck'].value_counts())
    CD_bef = data['Cabin_deck'].isna().sum()
    GCD_index = data[data['Cabin_deck'].isna()][(data[data['Cabin_deck'].isna()]['Group']).isin(GCD_gb.index)].index
    # Fill corresponding missing values
    data.loc[GCD_index, 'Cabin_deck'] = data.iloc[GCD_index, :]['Group'].map(lambda x: GCD_gb.idxmax(axis=1)[x])
    # Print number of missing values left
    # print('#Cabin_deck missing values before:', CD_bef)
    # print('#Cabin_deck missing values after:', data['Cabin_deck'].isna().sum())

print("7" * 100)
for data in data_cleaner:
    print(data[['HomePlanet', 'Destination', 'Solo', 'Cabin_deck']].value_counts())
    print(data.groupby(['HomePlanet', 'Destination', 'Solo', 'Cabin_deck'])['Cabin_deck'].size().unstack().fillna(0))
    CD_bef = data['Cabin_deck'].isna().sum()
    na_rows_CD = data.loc[data['Cabin_deck'].isna(), 'Cabin_deck'].index
    # data.loc[data['Cabin_deck'].isna(),'Cabin_deck']=data.groupby(['HomePlanet','Destination','Solo'])['Cabin_deck']
    test01 = data.groupby(['HomePlanet', 'Destination', 'Solo'])['Cabin_deck']
    data.loc[data['Cabin_deck'].isna(), 'Cabin_deck'] = test01.transform(lambda x: x.fillna(pd.Series.mode(x)[0]))[
        na_rows_CD]
    # print('#Cabin_deck missing values before:', CD_bef)
    # print('#Cabin_deck missing values after:', data['Cabin_deck'].isna().sum())

print("8" * 100)
# for data in data_cleaner:
#     plt.figure(figsize=(10, 4))
#     sns.scatterplot(x=data['Cabin_number'], y=data['Group'],
#                     c=LabelEncoder().fit_transform(data.loc[~data['Cabin_number'].isna(), 'Cabin_deck']), cmap='tab10')
#     plt.title('Cabin_number vs group colored by group')
# plt.show()
print('9' * 100)

for data in data_cleaner:
    CN_bef = data['Cabin_number'].isna().sum()
    print('#Cabin number missing values before:', CN_bef)
    for deck in ['A', 'B', 'C', 'D', 'E', 'F', 'G']:
        # X_CN = data.loc[~(data['Cabin_number'].isna()) & (data['Cabin_deck'] == deck), 'Group']
        # y_CN = data.loc[~(data['Cabin_number'].isna()) & (data['Cabin_deck'] == deck), 'Cabin_number']
        # X_test_CN = data.loc[(data['Cabin_number'].isna()) & (data['Cabin_deck'] == deck), 'Group']
        # if not X_test_CN.empty:
        #     model_CN = sklearn.linear_model.LinearRegression()
        #     model_CN.fit(X_CN.values.reshape(-1, 1), y_CN)
        #     preds_CN = model_CN.predict(X_test_CN.values.reshape(-1, 1))
        #     # Fillmissing values with predictions
        #     data.loc[(data['Cabin_number'].isna()) & (data['Cabin_deck'] == deck), 'Cabin_number'] = preds_CN.astype(
        #         int)
        X_CN = data.loc[(~data['Cabin_number'].isna()) & (data['Cabin_deck'] == deck), 'Group']
        y_CN = data.loc[(~data['Cabin_number'].isna()) & (data['Cabin_deck'] == deck), 'Cabin_number']
        X_test_CN = data.loc[(data['Cabin_number'].isna()) & (data['Cabin_deck'] == deck), 'Group']
        if not X_test_CN.empty:
            model_CN = sklearn.linear_model.LinearRegression()
            model_CN.fit(X_CN.values.reshape(-1, 1), y_CN)
            preds_CN = model_CN.predict(X_test_CN.values.reshape(-1, 1))
            # Fill missing values with predictions
            data.loc[(data['Cabin_number'].isna() & (data['Cabin_deck'] == deck)), 'Group'] = preds_CN

    # print('#Cabin number missing values before;', CN_bef)
    # print('#Cabin number missing values after;', data['Cabin_number'].isna().sum())

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
    # print('#VIP missing values after:', data['VIP'].isna().sum())
    # print('1' * 100)
for data in data_cleaner:
    # data.groupby(['HomePlanet', 'No_spending', 'Solo', 'Cabin_deck'])['Age'].median().unstack().fillna(0)
    # Missing values before
    A_bef = data[exp_feats].isna().sum().sum()
    # # Fill missing values using the median
    # na_rows_A = data.loc[data['Age'].isna(), 'Age'].index
    # data.loc[data['Age'].isna(), 'Age'] = \
    #     data.groupby(['HomePlanet', 'No_spending', 'Solo', 'Cabin_deck'])['Age'].transform(
    #         lambda x: x.fillna(x.median()))[na_rows_A]
    na_rows_A = data.loc[data['Age'].isna(), 'Age'].index
    data.loc[data['Age'].isna(), 'Age'] = \
        data.groupby(['HomePlanet', 'No_spending', 'Solo', 'Cabin_deck'])['Age'].transform(
            lambda x: x.fillna(x.median()))[
            na_rows_A]
    # # Print number of missing values left
    # print('#Age missing values before:', A_bef)
    # data.groupby(['HomePlanet', 'No_spending', 'Solo', 'Cabin_deck'])['Age'].transform(lambda x: x.fillna(x.median()))[
    #     na_rows_A]

print('2' * 100)
for data in data_cleaner:
    data.loc[data['Age'] <= 12, 'Age_group'] = 'Age_0-12'
    data.loc[(data['Age'] > 12) & (data['Age'] < 18), 'Age_group'] = 'Age_13_17'
    data.loc[(data['Age'] >= 18) & (data['Age'] <= 25), 'Age_group'] = 'Age_18-25'
    data.loc[(data['Age'] > 25) & (data['Age'] <= 30), 'Age_group'] = 'Age_26-30'
    data.loc[(data['Age'] > 30) & (data['Age'] <= 50), 'Age_group'] = 'Age_31-50'
    data.loc[data['Age'] > 50, 'Age_group'] = 'Age_51+'

for data in data_cleaner:
    # Join distribution
    data.groupby(['No_spending', 'CryoSleep'])['CryoSleep'].size().unstack().fillna(0)
    CSL_bef = data['CryoSleep'].isna().sum()
    # # Fill missing values using the mode
    na_rows_CSL = data.loc[data['CryoSleep'].isna(), 'CryoSleep'].index
    data.loc[data['CryoSleep'].isna(), 'CryoSleep'] = \
        data.groupby(['No_spending'])['CryoSleep'].transform(lambda x: x.fillna(pd.Series.mode(x)[0]))[na_rows_CSL]
    # print("#CryoSleep misisng values before:", CSL_bef)
    # print('#CryoSleep missing values after:', data['CryoSleep'].isna().sum())

for data in data_cleaner:
    # Missing value before
    E_bef = data[exp_feats].isna().sum().sum()
    for col in exp_feats:
        data.loc[(data[col].isna()) & (data['CryoSleep'] == True), col] = 0
    # print("#Expenditure missing values before", E_bef)
    # print("#Expenditure missing values after", data[exp_feats].isna().sum().sum())
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
    # data.isna().sum()
    for col in ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'Expenditure']:
        data[col] = np.log(1 + data[col])

## 3.23 Convert Formats
'''
## 3.23 Convert Formats

We will convert categorical data to dummy variables for mathematical analysis. There are multiple ways to encode categorical variables; we will use the sklearn and pandas functions.

In this step, we will also define our x (independent/features/explanatory/predictor/etc.) and y (dependent/target/outcome/response/etc.) variables for data modeling.

** Developer Documentation: **
* [Categorical Encoding](http://pbpython.com/categorical-encoding.html)
* [Sklearn LabelEncoder](http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html)
* [Sklearn OneHotEncoder](http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html)
* [Pandas Categorical dtype](https://pandas.pydata.org/pandas-docs/stable/categorical.html)
* [pandas.get_dummies](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.get_dummies.html)
'''
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
data1_x = ['HomePlanet', 'CryoSleep', 'Destination', 'Age', 'VIP', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa',
           'VRDeck']  # Original data
data1_x_calc = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'Expenditure', 'No_spending',
                'Group', 'Group_size', 'Solo', 'Cabin_number', 'Cabin_region1', 'Cabin_region2', 'Cabin_region3',
                'Cabin_region4', 'Cabin_region5', 'Cabin_region6', 'Cabin_region7', 'Family_size', 'HomePlanet_Code',
                'CryoSleep_Code', 'Destination_Code', 'VIP_Code', 'Age_group_Code', 'Cabin_deck_Code',
                'Cabin_side_Code']  # coded for algorithm calculation
data1_xy = Target + data1_x
print('Original X Y:', data1_xy, '\n')
# defina x variables for original w/bin features to remove continuous variables
data1_x_bin = ['Age', 'No_spending', 'Group_size', 'Solo', 'Cabin_region1', 'Cabin_region2', 'Cabin_region3',
               'Cabin_region4', 'Cabin_region5', 'Cabin_region6', 'Cabin_region7', 'Family_size', 'HomePlanet_Code',
               'CryoSleep_Code', 'Destination_Code', 'VIP_Code', 'Age_group_Code', 'Cabin_deck_Code', 'Cabin_side_Code']
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
train_x, test1_x, train1_y, test1_y = model_selection.train_test_split(data1[data1_x_calc], data1[Target],
                                                                       random_state=0)

train_x_bin, test1_x_bin, train1_y_bin, test1_y_bin = model_selection.train_test_split(data1[data1_x_bin],
                                                                                       data1[Target], random_state=0)

train1_x_dummy, test1_x_dummy, train1_y_dummy, test1_y_dummy = model_selection.train_test_split(
    data1_dummy[data1_x_dummy], data1[Target], random_state=0)

# print("Data1 Shape: {}".format(data1.shape))
# print("Train1 Shape: {}".format(train1_x.shape))
# print("Test1 Shape: {}".format(test1_x.shape))
print(train1_x_dummy.head())

for x in data1_x:
    if data1[x].dtype != 'float64':
        print('Transported Correlation by :', x)
        print(data1[[x, Target[0]]].groupby(x, as_index=False).mean())
        print('-' * 10, '\n')


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
    plt.title('Pearson correlation of features', y=1.05, size=15)


correlation_heatmap(data1)

print('6'*100)
#pair plots of entire dataset
pp = sns.pairplot(data1, hue = 'Transported', palette = 'deep', size=1.2, diag_kind = 'kde', diag_kws=dict(shade=True), plot_kws=dict(s=10) )
pp.set(xticklabels=[])
plt.show()


#Machine Learning Algorithm (MLA) Selection and Initialization
from sklearn import ensemble
MLA=[
    #Ensemble Methods
    ensemble.AdaBoostClassifier(),
    ensemble.BaggingClassifier(),

]
