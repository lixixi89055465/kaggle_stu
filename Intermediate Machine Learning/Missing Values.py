import pandas as pd
from sklearn.model_selection import train_test_split

# Load the data
data = pd.read_csv('../input/melbourne-housing-snapshot/melb_data.csv')

# Select target
y = data.Price
# To keep things simple, we'll use only numerical predictors
melb_predictors = data.drop(['Price'], axis=1)
X = melb_predictors.select_dtypes(exclude=['object'])
print(X.head())
print(X.columns)
X_train, X_valid, y_train, y_valid = \
    train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=0)
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error


def score_dataset(X_train, x_valid, y_train, y_valid):
    model = RandomForestRegressor(n_estimators=10, random_state=0)
    model.fit(X_train, y_train)
    preds = model.predict(x_valid)
    return mean_absolute_error(y_valid, preds)


# cols_with_misssing = [col for col in X_train.columns if X_train[col].isnull().any()]
print("1" * 100)
cols_with_missing = [col for col in X_train.columns if X_train[col].isnull().any()]
# Drop columns in training in trainnig
reduced_x_train = X_train.drop(cols_with_missing, axis=1)
reduced_x_valid = X_valid.drop(cols_with_missing, axis=1)
print("2" * 100)

print(score_dataset(reduced_x_train, reduced_x_valid, y_train, y_valid))
from sklearn.impute import SimpleImputer
