import pandas as pd
import numpy as np

df1 = pd.DataFrame([
    [178, '男', '深圳'],
    [183, '女', '广州'],
    [167, '男', '小红']],
    index=['小明', '小红', '小周'],
    columns=['身高', '性别', '地址'])
print('0' * 100)
print(df1)
df2 = df1.stack()
print('1' * 100)
print(df2)
print('2' * 100)
print(df2.index)
print(type(df1))
print(type(df2))

print('3' * 100)
print(df2)
data = pd.DataFrame(np.arange(12).reshape((3, 4)) + 100,
                    index=pd.Index(['date1', 'date2', 'date3']),
                    columns=pd.Index(['store1', 'store2', 'store3', 'store4']))
print(data)
print('4' * 100)
data2 = data.stack()
print(data2)
print('5' * 100)
print(data2.index)
print('6' * 100)
data3 = data2.unstack()
print(data3)
print('7' * 100)
col = pd.MultiIndex.from_tuples([('information', 'sex'), ('information', 'weight')])
print(col)
print('8' * 100)
df3 = pd.DataFrame([['男', 177], ['女', 168]],
                   index=['小明', '小红'],
                   columns=col)
print(df3)
print('9' * 100)
print(type(df3))
print(df3.index)
print(df3.columns)
df4 = df3.stack()
print('0' * 100)
print(df4)
print(df3)
print(df4.index)
print('1' * 100)
print(df3.columns)
print(df4.columns)
print('2' * 100)
print(df3)
print(df4)
print('3' * 100)
multicol2 = pd.MultiIndex.from_tuples([('weight', 'kg'),
                                       ('height', 'm'),
                                       ], name=['col', 'unit'])
data1 = pd.DataFrame([[1.0, 2.0],
                      [3.0, 4.0]],
                     index=['cat', 'dog'],
                     columns=multicol2)
print(data1)
print('4' * 100)
print(data1.columns)
print('5' * 100)
print(data1.stack())
print(data1.stack().index)
print('6' * 100)
print(data1.stack(1))
print('7' * 100)
print(data1.stack(1))
print('8' * 100)
print(data1.stack('unit'))
print('9' * 100)
print(data1.stack(0))
