import pandas as pd
from sklearn.model_selection import train_test_split

# Read the data
X_full = pd.read_csv('../input/home-data-for-ml-course/train.csv', index_col='Id')
X_test_full = pd.read_csv('../input/home-data-for-ml-course/test.csv', index_col='Id')

# Remove rows with missing target, separate target from predictors
X_full.dropna(axis=0, subset=['SalePrice'], inplace=True)
y = X_full.SalePrice
X_full.drop(['SalePrice'], axis=1, inplace=True)

# To keep things simple, we'll use only numerical predictors
X = X_full.select_dtypes(exclude=['object'])
X_test = X_test_full.select_dtypes(exclude=['object'])

# Break off validation set from training data
X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2,
                                                      random_state=0)
print(X_train.head())
print(X_train.columns)
print("1" * 100)
print(X_train.shape)
print("2" * 100)
missing_val_count_by_columns = X_train.isnull().sum()
print("3" * 100)
print(missing_val_count_by_columns[missing_val_count_by_columns > 0])
print("4" * 100)
print(X_train.shape)
print("5" * 100)
print(missing_val_count_by_columns.sum())
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error


def score_dataset(X_train, X_valid, y_train, y_valid):
    model = RandomForestRegressor(n_estimators=100, random_state=0)
    model.fit(X_train, y_train)
    preds = model.predict(X_valid)
    return mean_absolute_error(y_valid, preds)


cols_with_missing = [col for col in X_train.columns
                     if X_train[col].isnull().any()]

reduced_X_train = X_train.drop(cols_with_missing, axis=1)
reduced_X_valid = X_valid.drop(cols_with_missing, axis=1)
from sklearn.impute import SimpleImputer

myImputer = SimpleImputer()
imputed_X_train = pd.DataFrame(myImputer.fit_transform(X_train))
imputed_X_valid = pd.DataFrame(myImputer.transform(X_valid))

imputed_X_train.columns = X_train.columns
imputed_X_valid.columns = X_valid.columns
print("1" * 100)
print(imputed_X_train.head())
# final_imputer = SimpleImputer(strategy='mean')
# final_imputer = SimpleImputer(strategy='constant')
# final_imputer = SimpleImputer(strategy='most_frequent')
final_imputer = SimpleImputer(strategy='median')
final_X_train = pd.DataFrame(final_imputer.fit_transform(X_train))
final_X_valid = pd.DataFrame(final_imputer.transform(X_valid))
#
final_X_train.columns = X_train.columns
final_X_valid.columns = X_valid.columns
model = RandomForestRegressor(n_estimators=100, random_state=0)
model.fit(final_X_train, y_train)
preds_valid = model.predict(final_X_valid)
print('mae:')
print(mean_absolute_error(y_valid, y_pred=preds_valid))
# Fill in the line below: preprocess test data
final_X_test = pd.DataFrame(final_imputer.transform(X_test))

# Fill in the line below: get test predictions
preds_test = model.predict(final_X_test)

