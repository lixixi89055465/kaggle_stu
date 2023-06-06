import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
data = pd.read_csv('../input/fifa-2018-match-statistics/FIFA 2018 Statistics.csv')
print(data.columns)
print(data.head())
y = (data['Man of the Match'] == 'Yes')
feature_name = [i for i in data.columns if data[i].dtype in [np.int64]]
print(data.dtypes)
print(feature_name)
X = data[feature_name]
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)
my_model = RandomForestClassifier(n_estimators=100, random_state=0).fit(train_X, train_y)
import eli5
from eli5.sklearn import PermutationImportance

perm = PermutationImportance(estimator=my_model, random_state=1).fit(val_X, val_y)
eli5.show_weights(perm, feature_names=feature_name)
