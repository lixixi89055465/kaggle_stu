import pandas as pd
from sklearn.model_selection import train_test_split

# Read the input
data = pd.read_csv('../input/melbourne-housing-snapshot/melb_data.csv')

# Select subset of predictors
cols_to_use = ['Rooms', 'Distance', 'Landsize', 'BuildingArea', 'YearBuilt']
data.dropna(axis=0, inplace=True, subset=cols_to_use)
X = data[cols_to_use]

# Select target
y = data.Price

# Separate input into training and validation sets
X_train, X_valid, y_train, y_valid = train_test_split(X, y)

from xgboost import XGBRegressor

my_model = XGBRegressor()
my_model.fit(X_train, y_train)
from sklearn.metrics import mean_absolute_error

predictions = my_model.predict(X_valid)
print('mae:' + str(mean_absolute_error(y_valid, predictions)))
print('0' * 100)
my_model = XGBRegressor(n_estimators=500)
my_model.fit(X_train, y_train,
             # early_stopping_rounds=5,
             early_stopping_rounds=5,
             eval_set=[(X_valid, y_valid)],
             verbose=False)
print('1' * 100)
print(my_model.best_score)
my_model = XGBRegressor(n_estimators=1000, learning_rate=0.05)
my_model.fit(
    X_train, y_train, early_stopping_rounds=5,
    eval_set=[(X_valid, y_valid)],
    verbose=False
)
print(my_model.best_score)
print('2' * 100)
my_model = XGBRegressor(n_estimators=1000, learning_rate=0.05, n_jobs=4)
my_model.fit(
    X_train, y_train, early_stopping_rounds=5,
    eval_set=[(X_valid, y_valid)],
    verbose=False,
)
print(my_model.best_score)  
