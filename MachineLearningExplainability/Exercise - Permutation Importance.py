import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

data = pd.read_csv('../input/new-york-city-taxi-fare-prediction/train.csv', nrows=50000)
print(data.columns)
print(data.dtypes)
print('0' * 100)
data = data.query(
    'pickup_latitude > 40.7 and pickup_latitude < 40.8 and ' +
    'dropoff_latitude > 40.7 and dropoff_latitude < 40.8 and ' +
    'pickup_longitude > -74 and pickup_longitude < -73.9 and ' +
    'dropoff_longitude > -74 and dropoff_longitude < -73.9 and ' +
    'fare_amount > 0'
)
y = data.fare_amount
base_features = [
    'pickup_longitude',
    'pickup_latitude',
    'dropoff_longitude',
    'dropoff_latitude',
    'passenger_count'
]

X = data[base_features]
print('1' * 100)
print(X.head())
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)
first_model = RandomForestRegressor(n_estimators=50, random_state=1).fit(train_X, train_y)
print('2' * 100)
print(train_X.describe())
print('3' * 100)
print(train_y.describe())

import eli5
from eli5.sklearn import PermutationImportance

perm = PermutationImportance(estimator=first_model, random_state=1).fit(train_X, train_y)
eli5.show_weights(perm, feature_names=val_X.columns.tolist())
