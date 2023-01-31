import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score

df = pd.read_csv('./input/fe-course-input/concrete.csv')
print(df.head())
# X = df.copy()
# print(X.columns)
# y = X.pop('CompressiveStrength')
# baseline = RandomForestRegressor(criterion='mae', random_state=0)
# baseline_score = cross_val_score(baseline, X, y, cv=5, scoring='neg_mean_absolute_error')
# baseline_score = -1 * baseline_score.mean()
# print(f'MAX Base Line Score:{baseline_score:.4}')

'''
Index(['Cement', 'BlastFurnaceSlag', 'FlyAsh', 'Water', 'Superplasticizer',
       'CoarseAggregate', 'FineAggregate', 'Age', 'CompressiveStrength'],
       '''
X = df.copy()
y = X.pop('CompressiveStrength')

# Create synthetic features
X['FCRatio'] = X['FineAggregate'] / X['CoarseAggregate']
X['AggCmtRatio'] = (X['CoarseAggregate'] + X['FineAggregate']) / X['Cement']
X['WtrCmtRatio'] = X['Water'] / X['Cement']

model=RandomForestRegressor(criterion='mae',random_state=0)
from sklearn import metrics
score=cross_val_score(
    model,X,y,cv=5,scoring='neg_mean_absolute_error'
)
score=-1*score.mean()
print(f'MAX Score with Ratio Reatures :{score:.4}')

