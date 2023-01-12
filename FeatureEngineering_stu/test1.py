import pandas as pd
import numpy as np

df1 = pd.DataFrame({"key": list("abcd"), "data1": range(4)})
print("1" * 100)
print(df1)

df2 = pd.DataFrame({"key": ['a', 'b', 'b'], 'data2': range(3)})
print("2" * 100)
print(df2)

print("3" * 100)
df3 = df1.merge(df2)
print(df3)
print("4" * 100)
df4 = df1.merge(df2, left_on='key', right_on='key')
print(df4)
print("5" * 100)
df5 = pd.merge(df1, df2, left_on='key', right_on='key')
print(df5)
print("5.5" * 100)
df6 = pd.merge(df1, df2, how='outer')
print(df6)
print("7" * 100)
df7 = pd.merge(df1, df2, how='left')
print(df7)
df8=pd.merge(df1, df2, left_on='data1', right_index=True, suffixes=('_df1', '_df2'))
print("8" * 100)
print(df8)
