import pandas as pd
import numpy as np

df = pd.DataFrame([('bird', 2, 2),
                   ('mammal', 4, np.nan),
                   ('arthropod', 8, 0),
                   ('bird', 2, np.nan)],
                  index=('falcon', 'horse', 'spider', 'ostrich'),
                  columns=('species', 'legs', 'wings'))
print(df)

print('0' * 100)
print(df.mode())
print('1' * 100)
print(df.mode(dropna=False))
print('2' * 100)
print(df.mode(numeric_only=True))
print('3' * 100)
print(df.mode(axis='columns', numeric_only=True))
