import pandas as pd
from sklearn.model_selection import train_test_split

# Read the data
train_data = pd.read_csv('../input/home-data-for-ml-course/train.csv', index_col='Id')
test_data = pd.read_csv('../input/home-data-for-ml-course/test.csv', index_col='Id')
print(train_data.columns)
print(train_data.head())
y = train_data.SalePrice
train_data.drop(['SalePrice'], axis=1, inplace=True)
numerical_cols = [col for col in train_data.columns if train_data[col].dtype in ['int64', 'float64']]
X = train_data[numerical_cols].copy()

print(X.head())
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

my_pipeline = Pipeline(steps=[
    ('preprocessor', SimpleImputer()),
    ('model', RandomForestRegressor(n_estimators=50, random_state=0))
])

from sklearn.model_selection import cross_val_score

score = -1 * cross_val_score(my_pipeline, X, y, cv=5, scoring='neg_mean_absolute_error')
print('mae:')
print(score.mean())


def get_score(n_estimators):
    RandomForestRegressor(n_estimators, random_state=0)
    pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer()),
        ('model', RandomForestRegressor(n_estimators=n_estimators))
    ])
    scores = cross_val_score(pipeline, X=X, y=y, scoring='neg_mean_absolute_error', cv=3)
    return scores.mean()


print('0' * 100)
results = {}
for i in range(1, 9):
    results[50 * i] = get_score(i * 50)
print('1'*100)
import matplotlib.pyplot as plt

plt.plot(results.keys(), list(results.values()))
plt.show()
print(min(results, key=results.get))
