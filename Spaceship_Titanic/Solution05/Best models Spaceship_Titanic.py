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
fig = plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
sns.histplot(data=data1, x='Expenditure', hue='Transported', bins=200)
plt.title('Total expenditure (truncated) ')
plt.ylim([0, 200])
plt.xlim([0, 20000])
plt.subplot(1, 2, 2)
sns.countplot(data=data1, x='No_spending', hue='Transported')
plt.title('No spending indicator')
fig.tight_layout()
for dataset in data_cleaner:
    dataset['Expenditure'] = dataset[exp_feats].sum(axis=1)
    dataset['No_spending'] = (dataset['Expenditure'] == 0).astype(int)

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
plt.figure(figsize=(20,4))
plt.subplot(1,2,1)
sns.histplot(data=data1,x='Group',hue='PassengerId',binwidth=1)
plt.title('Group')
plt.subplot(1,2,2)
sns.countplot(data=data1,x='Group_size',hue='Transported')
plt.title('Group_size')
fig.tight_layout()
plt.show()
