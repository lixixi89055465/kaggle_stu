import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from xgboost import XGBRegressor
from sklearn.model_selection import cross_val_score

plt.style.use('seaborn-whitegrid')
plt.rc('figure', autolayout=True)
plt.rc(
    'axes',
    labelweight='bold',
    labelsize='large',
    titleweight='bold',
    titlesize=14,
    titlepad=10,
)


def apply_pca(X, standardize=True):
    if standardize:
        X = (X - X.mean(axis=0)) / X.std(axis=0)
    # Create principal components
    pca = PCA()
    X_pca = pca.fit_transform(X)
    component_names = [f"PC{i + 1}" for i in range(X_pca.shape[1])]
    X_pca = pd.DataFrame(X_pca, columns=component_names)
    loadings = pd.DataFrame(
        pca.components_.T,
        columns=component_names,
        index=X.columns,
    )
    return pca, X_pca, loadings


def score_dataset(X, y, model=XGBRegressor()):
    # Label encoding for categoricals
    for colname in X.select_dtypes(['category', 'object']):
        X[colname], = X[colname].factorize()
    model = XGBRegressor()
    score = cross_val_score(
        model, X, y, cv=5, scoring='neg_mean_squared_log_error',
    )
    score = -1 * score.mean()
    score = np.sqrt(score)
    return score


df = pd.read_csv('./input/fe-course-data/ames.csv')
feature = [
    "GarageArea",
    "YearRemodAdd",
    "TotalBsmtSF",
    "GrLivArea",
]
print(df.head())
print(df.columns)
print("Correlaction with SalePrice:\n")
print(df[feature].corrwith(df.SalePrice))

X = df.copy()
y = X.pop('SalePrice')
X = X.loc[:, feature]
pca, X_pca, loadings = apply_pca(X)
print(loadings)
X = X.join(X_pca)
score = score_dataset(X, y)
print(f"Your score :{score:.5f} RMSLE")
