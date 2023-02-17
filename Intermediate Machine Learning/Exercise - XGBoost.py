import pandas as pd
from sklearn.model_selection import train_test_split

# Read the data
X = pd.read_csv('../input/melbourne-housing-snapshot/train.csv', index_col='Id')
X_test_full = pd.read_csv('../input/melbourne-housing-snapshot/test.csv', index_col='Id')

X.dropna(axis=0, subset=['SalePrice'], inplace=True)
y = X.SalePrice
X.drop(['SalePrice'], axis=1, inplace=True)
X_train_full, X_valid_full, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=0)
low_cardinality_cols = [cname for cname in X_train_full.columns
                        if (X_train_full[cname].dtype == 'object' and
                            X_train_full[cname].nunique() < 10)]
numerical_cols = [cname for cname in X_train_full.columns
                  if (X_train_full[cname].dtype in ['int64', 'float64'])]

my_cols = low_cardinality_cols + numerical_cols
X_train = X_train_full[my_cols].copy()
X_valid = X_valid_full[my_cols].copy()
X_test = X_test_full[my_cols].copy()

X_train = pd.get_dummies(X_train)
X_valid = pd.get_dummies(X_valid)
X_test = pd.get_dummies(X_test)
print("0" * 100)
X_train, X_valid = X_train.align(X_valid, join='left', axis=1)
print(X_train.head())
X_train, X_test = X_train.align(X_test, join='left', axis=1)

from xgboost import XGBRegressor

# Define the model
my_model_1 = XGBRegressor(random_state=0)
# Fit the model
my_model_1.fit(X_train, y_train)  # Your code here

from sklearn.metrics import mean_absolute_error

# Get predictions
predictions_1 = my_model_1.predict(X_valid)  # Your code here
# Calculate MAE
mae_1 = mean_absolute_error(y_valid, predictions_1)  # Your code here

# Uncomment to print MAE
# print("Mean Absolute Error:" , mae_1)
my_model_2 = XGBRegressor(n_estimators=10, learning_rate=0.1)
my_model_2.fit(X_train, y_train)
predictions_2 = my_model_2.predict(X_valid)
mae_2 = mean_absolute_error(y_valid, predictions_2)
print("1" * 100)
print(mae_1)
print(mae_2)



