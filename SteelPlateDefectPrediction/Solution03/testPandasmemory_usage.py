# -*- coding: utf-8 -*-
# @Time    : 2024/5/24 下午11:12
# @Author  : nanji
# @Site    : 
# @File    : testPandasmemory_usage.py
# @Software: PyCharm 
# @Comment :
import pandas as pd
import numpy as np

dtypes = ['int64', 'float64', 'complex128', 'object', 'bool']
data = dict([(t, np.ones(shape=5000).astype(t)) for t in dtypes])

df = pd.DataFrame(data)
print(df.head())
print('0' * 100)
print(df.memory_usage())

print('1' * 100)
print(df.memory_usage())

print('2' * 100)
r1 = df.memory_usage(index=False)
print(r1)

print('3' * 100)
r2 = df.memory_usage(deep=True)
print(r2)
r3 = df['object'].astype('category').memory_usage(deep=True)
print('4' * 100)
print(r3)
