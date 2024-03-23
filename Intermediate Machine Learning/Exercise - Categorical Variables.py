import pandas as pd
from sklearn.model_selection import train_test_split

X = pd.read_csv('../input/home-input-for-ml-course/train.csv', index_col='Id')
X_test = pd.read_csv('../input/home-input-for-ml-course/test.csv', index_col='Id')
X.dropna(axis=0, subset=['SalePrice'], inplace=True)
y = X.SalePrice
X.drop(['SalePrice'], axis=1, inplace=True)
cols_with_missing = [col for col in X.columns if X[col].isnull().any()]
X.drop(cols_with_missing, axis=1, inplace=True)
X_test.drop(cols_with_missing, axis=1, inplace=True)
X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=0)

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error


def score_dataset(X_train, X_valid, y_train, y_valid):
    model = RandomForestRegressor(n_estimators=100, random_state=0)
    model.fit(X_train, y_train)
    preds = model.predict(X_valid)
    return mean_absolute_error(y_valid, preds)


# Categorical columns in the training input
object_cols = [col for col in X_train.columns if X_train[col].dtype == "object"]

# Columns that can be safely ordinal encoded
good_label_cols = [col for col in object_cols if
                   set(X_valid[col]).issubset(set(X_train[col]))]

# Problematic columns that will be dropped from the dataset
bad_label_cols = list(set(object_cols) - set(good_label_cols))
object_nunique = list(map(lambda col: X_train[col].nunique(), object_cols))

drop_X_train = X_train.select_dtypes(exclude=['object'])
drop_X_valid = X_valid.select_dtypes(exclude=['object'])
low_cardinality_cols = [col for col in object_cols
                        if X_train[col].nunique() < 10]
high_cardinality_cols = list(set(object_cols) - set(low_cardinality_cols))
print("0" * 100)
print(low_cardinality_cols)
print("1" * 100)
print(high_cardinality_cols)
from sklearn.preprocessing import OneHotEncoder

myOneHotEncoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
OH_cols_train = pd.DataFrame(myOneHotEncoder.fit_transform(X_train[low_cardinality_cols]))
OH_cols_valid = pd.DataFrame(myOneHotEncoder.transform(X_valid[low_cardinality_cols]))
OH_cols_train.index = X_train.index
OH_cols_valid.index = X_valid.index

num_x_train = X_train.drop(object_cols, axis=1)
num_x_valid = X_valid.drop(object_cols, axis=1)
OH_X_train = pd.concat([num_x_train, OH_cols_train], axis=1)
OH_X_valid = pd.concat([num_x_valid, OH_cols_valid], axis=1)
print("3" * 100)
print(score_dataset(OH_X_train, OH_X_valid, y_train, y_valid))
