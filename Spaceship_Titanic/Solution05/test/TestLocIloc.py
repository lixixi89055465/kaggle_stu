import pandas as pd
import numpy as np

df = pd.DataFrame([[1, 2], [4, 5], [7, 8]],
                  index=['cobra', 'viper', 'sidewinder'],
                  columns=['max_speed', 'shield'])
print(df)
print("0" * 100)
print(df.loc['viper'])
print("2" * 100)
print(df.loc[['viper', 'sidewinder']])
print(df.loc['viper', 'shield'])

print("1" * 100)
print(df.loc[[False, False, True]])
print("3" * 100)
print(df.loc[pd.Series([False, True, False],
                       index=['cobra', 'viper', 'sidewinder'])])

print(df.loc[pd.Index(['cobra', 'viper'], name='foo')])
print("4" * 100)
print(df.loc[df['shield'] > 6])
print("5" * 100)
print(df.loc[df['shield'] > 6, ['max_speed']])
print(df.loc[lambda df: df['shield'] == 8])
df.loc[['viper', 'sidewinder'], ['shield']] = 50
print("6" * 100)
print(df)
df.loc['cobra'] = 10
print("7" * 100)
print(df)
df.loc[:, 'max_speed'] = 30
print("8" * 100)
print(df)
df.loc[df['shield'] > 35] = 0
print(df)
df = pd.DataFrame([[1, 2], [4, 5], [7, 8]],
                  index=[7, 8, 9], columns=['max_speed', 'shield'])
print("9" * 100)
print(df)
print("0" * 100)
print(df.loc[7:9])

tuples = [
    ('cobra', 'mark i'), ('cobra', 'mark ii'),
    ('sidewinder', 'mark i'), ('sidewinder', 'mark ii'),
    ('viper', 'mark ii'), ('viper', 'mark iii')
]
index = pd.MultiIndex.from_tuples(tuples)
values = [[12, 2], [0, 4], [10, 20],
          [1, 4], [7, 1], [16, 36]]
print("2" * 100)
df = pd.DataFrame(values, columns=['max_speed', 'shield'], index=index)
print(df)

print("3" * 100)
print(df.loc['cobra', 'shield'])
print("4" * 100)
print(df.loc['cobra':'viper', 'max_speed'])
print("5" * 100)
print(df)
# print(df.loc[[False, False, True]])
# print("6" * 100)
# print(df.loc[pd.Series([False, True, False],
#                        index=['viper', 'sidewinder', 'cobra'])])
# df.loc[pd.Series([False, True, False],
#        index=['viper', 'sidewinder', 'cobra'])]
print("6" * 100)
print(df.loc[pd.Index(['cobra', 'viper'], name='foo')])
print("7" * 100)
tuples = [
    ('cobra', 'mark i'), ('cobra', 'mark ii'),
    ('sidewinder', 'mark i'), ('sidewinder', 'mark ii'),
    ('viper', 'mark ii'), ('viper', 'mark iii')
]
index = pd.MultiIndex.from_tuples(tuples)
values = [[12, 2], [0, 4], [10, 20],
          [1, 4], [7, 1], [16, 36]]
df = pd.DataFrame(values, columns=['max_speed', 'shield'], index=index)
print(df)
print("8" * 100)
print(df.loc['cobra', 'mark i'])
print(df.loc[[('cobra', 'mark ii')]])
print("9" * 100)
print(df.loc[('cobra', 'mark i'), 'shield'])
print("0" * 100)
print(df.loc[('cobra', 'mark i'):'viper'])
print("1" * 100)
print(df.loc[('cobra', 'mark ii'):('viper', 'mark ii')])

