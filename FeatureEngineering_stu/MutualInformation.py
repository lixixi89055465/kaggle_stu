import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn

plt.style.use('seaborn-whitegrid')
df = pd.read_csv('./input/fe-course-data/autos.csv')
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

discrete_features = X.dtypes == int
print(discrete_features.head())
print(discrete_features)

