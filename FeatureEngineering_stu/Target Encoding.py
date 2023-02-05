import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import warnings
from sklearn.model_selection import cross_val_score
from xgboost import XGBRegressor
from category_encoders import MEstimateEncoder

# Set matplotlib defaults
plt.style.use('seaborn-whitegrid')
plt.rc("figure", autolayout=True)
plt.rc(
    "axes",
    labelweight='bold',
    labelsize='large',
    titleweight='bold',
    titlesize=14,
    titlepad=10,
)
warnings.filterwarnings('ignore')


def score_dataset(X, y, model=XGBRegressor()):
    for colname in X.select_dtypes(['category', 'object']):
        X[colname], _ = X[colname].factorize()
    score = cross_val_score(
        model, X, y, cv=5, scoring='neg_mean_squared_log_err',
    )
    score = -1 * score.mean()
    score.np.sqrt(score)
    return score


df = pd.read_csv('./input/fe-course-data/ames.csv')
# print(df.head())
print('1' * 100)
print(df.select_dtypes(['object']).nunique())
print('2' * 100)
print(df['SaleType'].value_counts())
print('3' * 100)
# Encoding split
print(df.shape)
X_encode = df.sample(frac=0.2, random_state=0)
y_encode = X_encode.pop('SalePrice')
# Training split
X_pretrain = df.drop(index=X_encode.index)
print(X_pretrain.shape)
