import pandas as pd
import numpy as np

s = pd.Series(list('abca'))
print(pd.get_dummies(s))
s1 = ['a', 'b', np.nan]
print('0' * 100)
print(pd.get_dummies(s1))

print('1' * 100)
df = pd.DataFrame({'A': ['a', 'b', 'a'], 'B': ['b', 'a', 'c'],
                   'C': [1, 2, 3]})
print(df)
print('2' * 100)
print(pd.get_dummies(df))
print('3' * 100)
df = pd.DataFrame({'key': ['b', 'b', 'a', 'c', 'a', 'b'], 'data1': np.arange(6)})
print(pd.get_dummies(df['key']))
dum_key = pd.get_dummies(df['key'], prefix='key')
print('4'*100)
print(df[['data1']].join(dum_key))
