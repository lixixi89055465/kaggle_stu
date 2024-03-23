import pandas as pd
from sklearn.model_selection import train_test_split

# Load the input
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

my_imuter = SimpleImputer()
imputed_X_train = pd.DataFrame(my_imuter.fit_transform(X_train))
imputed_X_valid = pd.DataFrame(my_imuter.fit_transform(X_valid))

imputed_X_train.columns = X_train.columns
imputed_X_valid.columns = X_valid.columns
print("3" * 100)
print(score_dataset(imputed_X_train, imputed_X_valid, y_train, y_valid))
print("4" * 100)
print(X_train.shape)
missing_val_count_by_column = (X_train.isnull().sum())

X_train_plus = X_train.copy()
X_valid_plus = X_valid.copy()
print("5" * 100)
for col in cols_with_missing:
    X_train_plus[col + '_was_missing'] = X_train_plus[col].isnull()
    X_valid_plus[col + '_was_missing'] = X_valid_plus[col].isnull()

my_imuter = SimpleImputer()
imputed_X_train_plus = pd.DataFrame(my_imuter.fit_transform(X_train_plus))
imputed_X_valid_plus = pd.DataFrame(my_imuter.transform(X_valid_plus))
print("6" * 100)

print(imputed_X_train_plus.head())
imputed_X_train_plus.columns = X_train_plus.columns
imputed_X_valid_plus.columns = X_valid_plus.columns
print("7" * 100)
print(score_dataset(imputed_X_train_plus, imputed_X_valid_plus, y_train, y_valid))

# Shape of training input (num_rows, num_columns)
print(X_train.shape)
missing_val_count_by_column = (X_train.isnull().sum())
print("8"*100)
print(missing_val_count_by_column[missing_val_count_by_column > 0])
