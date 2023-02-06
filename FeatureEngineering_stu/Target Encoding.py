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
        model, X, y, cv=5, scoring='neg_mean_squared_log_error',
    )
    score = -1 * score.mean()
    score = np.sqrt(score)
    return score


df = pd.read_csv('./input/fe-course-input/ames.csv')
print(df.select_dtypes(['object']).nunique())
print(df['SaleType'].value_counts())
# Encoding split
X_encode = df.sample(frac=0.2, random_state=0)
y_encode = X_encode.pop('SalePrice')
# Training split
X_pretrain = df.drop(index=X_encode.index)
y_train = X_pretrain.pop("SalePrice")
print(X_pretrain.shape)

# encoder = MEstimateEncoder()
encoder = MEstimateEncoder(cols=['Neighborhood'], m=1.0)
encoder.fit_transform(X_encode, y_encode)
X_train = encoder.transform(X_pretrain, y_train)
print('2' * 100)
feature = encoder.cols
print(feature)
plt.figure(dpi=90)
ax = sns.distplot(y_train, kde=True, hist=True)
# ax = sns.displot(x=y_train, color='b')
ax = sns.distplot(X_train[feature], color='r', ax=ax, hist=True, kde=False, norm_hist=True)
# ax = sns.displot(input=X_train, x=feature[0], color='r', ax=ax, hue_norm=True, kind='hist')

# ax.set_xlabels("SalePrice")
ax.set_xlabel("SalePrice")
plt.show()
X = df.copy()
y = X.pop('SalePrice')
score_base = score_dataset(X, y)
score_new = score_dataset(X_train, y_train)
print(f"Baseline Score:{score_base:.4f} RMSLE")
print(f"Score with Encoding :{score_new:.4f} RMSLE")
# Try experimenting with the smoothing parameter m
# Try 0, 1, 5, 50
m =5
X = df.copy()
y = X.pop('SalePrice')
# Creat an uniformative feature
X['Count'] = range(len(X))
X['Count'][1] = 0  # actually need one duplicate value to circumvent
# Fit and transform on the same dataset

encoder = MEstimateEncoder(cols='Count', m=m)
X = encoder.fit_transform(X, y)
# Results
score = score_dataset(X, y)
print(f'Score:{score:.4f} RMSLE')

print('3' * 100)
plt.figure(dpi=90)
ax = sns.distplot(y, kde=True, hist=False)
ax = sns.distplot(X['Count'], color='r', ax=ax, hist=True,
                  kde=False, norm_hist=True)
ax.set_xlabel('SalePrice')
plt.show()
print('3'*100)
X = df.copy()
y = X.pop("SalePrice")
score_base = score_dataset(X, y)
score_new = score_dataset(X_train, y_train)

print(f"Baseline Score: {score_base:.4f} RMSLE")
print(f"Score with Encoding: {score_new:.4f} RMSLE")