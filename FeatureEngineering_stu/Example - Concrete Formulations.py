import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score

df=pd.read_csv('./input/fe-course-data/concrete.csv')
print(df.head())
