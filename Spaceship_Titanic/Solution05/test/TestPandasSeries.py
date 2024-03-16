import pandas as pd
import numpy as np

s = pd.Series(['X', 'X', 'Y', 'X'])
print(s)
# 0    X
# 1    X
# 2    Y
# 3    X
# dtype: object

print(s.mode())
# 0    X
# dtype: object

print(type(s.mode()))
# <class 'pandas.core.series.Series'>

mode_value = s.mode()[0]
print('0' * 100)
print(mode_value)
s_same = pd.Series(['X', 'Y', 'Y', 'X'])
print('1' * 100)
print(s_same)
print(s_same.mode()[0])
print(s_same.mode()[1])
l_modes = s_same.mode().tolist()
print('2'*100)

df = pd.DataFrame({'col1': ['X', 'Y', 'Y', 'X'],
                   'col2': ['X', 'Y', 'Y', 'X']},
                  index=['row1', 'row2', 'row3', 'row4'])
print(df)
print('3'*100)
print(list(df.mode()))
print('4'*100)
print(df.mode())
print('5'*100)
print(type(df.mode()))

print('6'*100)
print(df.mode().count())

print('7'*100)
print(df.mode().iloc[0])
print('8'*100)
print(df.apply(lambda x: x.mode().tolist()))
