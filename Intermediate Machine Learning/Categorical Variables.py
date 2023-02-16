import pandas as pd
from sklearn.model_selection import train_test_split

data = pd.read_csv('../input/melbourne-housing-snapshot/melb_data.csv')

print(data.head())
print(data.columns)
#
y = data.Price
x = data.drop(['Price'], axis=1)
X_train_full, X_valid_full, y_train, y_valid = train_test_split(x, y, train_size=0.8, test_size=0.2, random_state=0)
cols_with_missing = [col for col in X_train_full.columns if X_train_full[col].isnull().any()]
print(cols_with_missing)

X_train_full.drop(cols_with_missing, axis=1, inplace=True)
X_valid_full.drop(cols_with_missing, axis=1, inplace=True)
low_cardinality_cols = [cname for cname in X_train_full.columns
                        if X_train_full[cname].nunique() < 10 and X_train_full[cname].dtype == 'object']

low_cardinality_cols = [cname for cname in X_train_full.columns
                        if X_train_full[cname].nunique() < 10 and
                        X_train_full[cname].dtype == 'object']
numberical_cols = [cname for cname in X_train_full.columns
                   if X_train_full[cname].dtype in ['int64', 'float64']]

my_cols = low_cardinality_cols + numberical_cols
X_train = X_train_full[my_cols].copy()
X_valid = X_valid_full[my_cols].copy()
print("3" * 100)
print(X_train.head())
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error


def score_dataset(X_train, X_valid, y_train, y_valid):
    model = RandomForestRegressor(n_estimators=100, random_state=0)
    model.fit(X_train, y_train)
    preds = model.predict(X_valid)
    return mean_absolute_error(y_valid, preds)


drop_X_train = X_train.drop()
drop_X_valid = ____

