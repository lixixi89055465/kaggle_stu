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


# Load data
df = pd.read_csv("./input/fe-course-data/ames.csv")

features = ["YearBuilt", "MoSold", "ScreenPorch"]
sns.relplot(x='value', y='SalePrice', col='variable', data=df.melt(id_vars='SalePrice', value_vars=features),
            facet_kws=dict(sharex=False), )
