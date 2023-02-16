import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.model_selection import cross_val_score
from xgboost import XGBRegressor

plt.style.use("seaborn-whitegrid")
plt.rc('figure', autolayout=True)
plt.rc(
    'axes',
    labelweight='bold',
    labelsize='large',
    titleweight='bold',
    titlesize=14,
    titlepad=10,
)

df = pd.read_csv('./input/fe-course-input/ames.csv')
print(df.head())
print(df.columns)
X = df.copy()
y = X.pop('SalePrice')
features = ['LotArea', 'TotalBsmtSF', 'FirstFlrSF', 'SecondFlrSF', 'GrLivArea']
X_scaled = X.loc[:, features]
X_scaled = (X_scaled - X_scaled.mean(axis=0)) / X_scaled.std(axis=0)
kmeans = KMeans(n_clusters=10, random_state=0)
X['Cluster'] = kmeans.fit_predict(X_scaled)
print(X.head())
Xy = X.copy()
Xy['Cluster'] = Xy.Cluster.astype('category')
Xy['SalePrice'] = y
sns.relplot(
    x='value', y='SalePrice', hue='Cluster', col='variable',
    height=4, aspect=1, facet_kws={'sharex': False}, col_wrap=3,
    data=Xy.melt(
        value_vars=features, id_vars=['SalePrice', 'Cluster'],
    )
)


def score_dataset(X, y, model=XGBRegressor()):
    # Label encoding for categoricals
    for colname in X.select_dtypes(['category', 'object']):
        X[colname], _ = X[colname].factorize()
    # Metric for Housing competition is RMSLE (Root Mean Squared Log Error)
    score = cross_val_score(
        model, X, y, cv=5, scoring='neg_mean_squared_log_error'
    )
    score = -1 * score.mean()
    score = np.sqrt(score)
    return score


score_dataset(X, y)
plt.show()
print('2' * 100)
kmeans = KMeans(n_clusters=10, n_init=10, random_state=0)
X_cd = kmeans.fit_transform(X_scaled)
# Label features and join to dataset
X_cd = pd.DataFrame(X_cd, columns=[f"Centroid_{i}" for i in range(X_cd.shape[1])])
X = X.join(X_cd)
print(X.head())
