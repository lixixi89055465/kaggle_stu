# import basic libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error

# Big Mart Sales
## preprocessing
### mean imputations
# Big Mart Sales
train = pd.read_csv("../input/bigmart-sales-data/Train.csv")
test = pd.read_csv("../input/bigmart-sales-data/Test.csv")
print(train.columns)
print(train.shape)
print(train.head())
print("0" * 100)
train['Item_Weight'].fillna((train['Item_Weight'].mean()), inplace=True)
test['Item_Weight'].fillna((test['Item_Weight'].mean()), inplace=True)

# Big Mart Sales

### reducing fat content to only two categories
number = preprocessing.LabelEncoder()
# number = LabelEncoder()
train['Item_Fat_Content'] = train['Item_Fat_Content'].replace(['low fat', 'LF'], ['Low Fat', 'Low Fat'])
train['Item_Fat_Content'] = train['Item_Fat_Content'].replace(['reg'], ['Regular'])
test['Item_Fat_Content'] = test['Item_Fat_Content'].replace(['low fat', 'LF'], ['Low Fat', 'Low Fat'])
test['Item_Fat_Content'] = test['Item_Fat_Content'].replace(['reg'], ['Regular'])
train['Outlet_Establishment_Year'] = 2013 - train['Outlet_Establishment_Year']
test['Outlet_Establishment_Year'] = 2013 - test['Outlet_Establishment_Year']

train['Outlet_Size'].fillna('Small', inplace=True)
test['Outlet_Size'].fillna('Small', inplace=True)

train['Item_Visibility'] = np.sqrt(train['Item_Visibility'])
test['Item_Visibility'] = np.sqrt(test['Item_Visibility'])

col = ['Outlet_Size', 'Outlet_Location_Type', 'Item_Fat_Content']
test['Item_Outlet_Sales'] = 0

combi = train.append(test)
print("2"*100)
print(combi.info())
print("3"*100)
for i in col:
    combi = number.fit_transform(combi.astype('str'))
    combi = combi.astype('object')

train = combi[:train.shape[0]]
test = combi[train.shape[0]:]
test.drop('Item_Outlet_Sales', axis=1, inplace=True)
##
from sklearn.model_selection import train_test_split

tpot_train = train.drop(['Outlet_Identifier', 'Item_type', 'Item_Identifier'], axis=1)
tpot_test = test.drop(['Outlet_Identifier', 'Item_type', 'Item_Identifier'], axis=1)
target = tpot_train['Item_Outlet_Sales']
tpot_train.drop('Item_Outlet_Sales', axis=1, inplace=True)
# finally building model using tpot library

from tpot import TPOTRegressor
