import pandas as pd
import numpy as np

t1 = pd.Series([False, False]).any()
print(t1)
t2 = pd.Series([True, False]).any()
print(t2)

t3 = pd.Series([]).any()
print(t3)
t4 = pd.Series([np.nan]).any()
print(t4)
t5 = pd.Series([np.nan]).any(skipna=False)
print(t5)

t6 = pd.Series([False, False]).any()
print(t6)
t7 = pd.Series([True, True]).all()
print(t7)
t8 = pd.Series([True, False]).all()
print(t8)
print('0' * 100)
t9 = pd.Series([np.nan]).all()
print(t9)
df = pd.DataFrame({'col1': [True, True], 'col2': [True, False]})
print(df)
print('1'*100)
print(df.all())
print('2'*100)
print(df.all(1))
print('3'*100)
print(df.all(axis=None))
