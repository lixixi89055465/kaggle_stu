import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA

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
    # convert to dataframe
    component_names = [f"PC{i + 1}" for i in range(X_pca.shape[1])]
    X_pca = pd.DataFrame(X_pca, columns=component_names)
    # Create loadings
    loadings = pd.DataFrame(
        pca.components_.T,  # transpose the matrix of loadings
        columns=component_names,  # so the columns are the  principal components
        index=X.columns,
    )
    return pca, X_pca, loadings

    pass


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
