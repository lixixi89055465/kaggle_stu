import pandas as pd
from sklearn.model_selection import train_test_split

# Read the input
X_full = pd.read_csv('../input/home-input-for-ml-course/train.csv', index_col='Id')
X_test_full = pd.read_csv('../input/home-input-for-ml-course/test.csv', index_col='Id')

# Obtain target and predictors
y = X_full.SalePrice
features = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']
X = X_full[features].copy()
X_test = X_test_full[features].copy()

# Break off validation set from training input
X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2,
                                                      random_state=0)

print(X_train.head())
from sklearn.ensemble import RandomForestRegressor

model_1 = RandomForestRegressor(n_estimators=50, random_state=0)
model_2 = RandomForestRegressor(n_estimators=100, random_state=0)
model_3 = RandomForestRegressor(n_estimators=100, criterion='absolute_error', random_state=0)
model_4 = RandomForestRegressor(n_estimators=50, random_state=0, min_samples_split=20)
model_5 = RandomForestRegressor(n_estimators=50, random_state=0, max_depth=7)
models = [model_1, model_2, model_3, model_4, model_5]

from sklearn.metrics import mean_absolute_error


def score_model(model, X_t=X_train, X_v=X_valid, y_t=y_train, y_v=y_valid):
    model.fit(X_train, y_train)
    preds = model.predict(X_v)
    return mean_absolute_error(y_v, preds)


for i in range(0, len(models)):
    mae = score_model(models[i], X_train, X_valid, y_train, y_valid)
    print('Model %d MAE: %d' % (i + 1, mae))

print("2" * 100)
my_model = model_3
my_model.fit(X, y)
preds_test = my_model.predict(X_test)
output = pd.DataFrame({'Id': X_test.index,
                       'SalePrice': preds_test})
output.to_csv('submission.csv', index=False)
