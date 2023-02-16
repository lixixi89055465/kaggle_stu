import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.feature_selection import mutual_info_regression

# Set Matplotlib defaults
plt.style.use("seaborn-whitegrid")
plt.rc("figure", autolayout=True)
plt.rc(
    "axes",
    labelweight="bold",
    labelsize="large",
    titleweight="bold",
    titlesize=14,
    titlepad=10,
)


def make_mi_scores(X, y):
    X = X.copy()
    for colname in X.select_dtypes(['object']):
        X[colname], _ = X[colname].factorize()
    # All discrete features should now have integer dtypes
    discrete_features = [pd.api.types.is_integer_dtype(t) for t in X.dtypes]
    print(discrete_features)


# Load input
df = pd.read_csv("./input/fe-course-input/ames.csv")

features = ["YearBuilt", "MoSold", "ScreenPorch"]
# sns.relplot(x='value', y='SalePrice', col='variable', input=df.melt(id_vars='SalePrice', value_vars=features),
#             facet_kws=dict(sharex=False), )
def make_mi_scores(X, y, discrete_features):
    mi_scores = mutual_info_regression(X, y, discrete_features=discrete_features)
    mi_scores = pd.Series(mi_scores, name='MI Scores', index=X.columns)
    print(mi_scores)
    mi_scores = mi_scores.sort_values(ascending=False)
    return mi_scores

X=df.copy()
y=X.pop('SalePrice')
for colname in X.select_dtypes("object"):
    X[colname],_=X[colname].factorize()
discrete_features= X.dtypes==int
make_mi_scores(X,y,discrete_features)

