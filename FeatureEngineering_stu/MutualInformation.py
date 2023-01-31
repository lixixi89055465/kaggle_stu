import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

plt.style.use('seaborn-whitegrid')
df = pd.read_csv('./input/fe-course-input/autos.csv')
print(df.head())

X = df.copy()
y = X.pop('price')

# Label encoding for categories
print(X.dtypes)
# for colname in X.select_dtypes(include='object'):
#     X[colname],_=X[colname].factorize()
print('1' * 100)
print(X.select_dtypes(include='object'))
print('2' * 100)
for colname in X.select_dtypes('object'):
    X[colname], _ = X[colname].factorize()
print('3' * 100)
print(X.head())
discrete_features = X.dtypes == int
from sklearn.feature_selection import mutual_info_regression

print('4' * 100)


def make_mi_scores(X, y, discrete_features):
    mi_scores = mutual_info_regression(X, y, discrete_features=discrete_features)
    mi_scores = pd.Series(mi_scores, name='MI Scores', index=X.columns)
    print(mi_scores)
    mi_scores = mi_scores.sort_values(ascending=False)
    return mi_scores


mis_scores = make_mi_scores(X, y, discrete_features)
print(mis_scores[::3])



def plot_mi_scores(scores):
    scores = scores.sort_values(ascending=True)
    width = np.arange(len(scores))
    ticks = list(scores.index)
    plt.barh(width, scores)
    plt.yticks(width, ticks)
    plt.title('Multual Information Scores')

#
# plt.figure(dpi=100, figsize=(8, 5))
# plot_mi_scores(scores=mis_scores)
sns.relplot(x='curb_weight',y='price',data=df)

