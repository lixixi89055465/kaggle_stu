import pandas as pd

mydict = [{'a': 1, 'b': 2, 'c': 3, 'd': 4},
          {'a': 100, 'b': 200, 'c': 300, 'd': 400},
          {'a': 1000, 'b': 2000, 'c': 3000, 'd': 4000}]
df = pd.DataFrame(mydict)
print(df)
print('1' * 40)
print(type(df.iloc[0]))
print(df.iloc[:3])
print('2' * 40)
print(df.iloc[[True, False, True]])
print('3' * 40)
print(df.iloc[lambda x: x.index % 2 == 0])
print('4' * 40)
print(df.iloc[0, 1])
print('5' * 40)
print(df.iloc[1:3, 0:3])

