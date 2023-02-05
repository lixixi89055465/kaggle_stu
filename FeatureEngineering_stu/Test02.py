# -*- coding: utf-8 -*
from sklearn import datasets, linear_model
from sklearn.model_selection import cross_val_score
import pandas as pd

df = pd.DataFrame({'A': [1, 2, 3], 'B': [1, 1, 1]})
print(df)
print('1'*100)
print(df.nunique())
