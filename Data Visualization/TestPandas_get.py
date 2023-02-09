import numpy as np
from pandas import DataFrame
import pandas as pd

df = DataFrame(np.arange(12).reshape((3, 4)), index=['one', 'two', 'thr'], columns=list('abcd'))
print(df)
print(df[['a', 'b']])

print(df.loc['one', 'a'])
print('1' * 100)
print(df.loc['one':'two', 'a':'b'])
print('2' * 100)

print(df.loc['one':'two', ['a', 'c']])
# loc 只能通过index和 columns 来取，不能用数字


print('3' * 100)
# iloc只能用数字索引，不能用索引名
print(df.iloc[0:2])
print('4' * 100)
print(df.iloc[0])
print('5' * 100)
print(df.iloc[0:2, 0:2])
df.iloc[[0, 2], [1, 2, 3]]
print('6' * 100)

print(df.iat[1, 1])
print('7' * 100)

print(df.at['one', 'a'])
