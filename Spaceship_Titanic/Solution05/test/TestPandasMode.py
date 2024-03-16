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
print('4' * 100)
# importing pandas as pd
import pandas as pd

# Creating the dataframe
df = pd.DataFrame({"A": [14, 4, 5, 4, 1],
                   "B": [5, 2, 54, 3, 2],
                   "C": [20, 20, 7, 3, 8],
                   "D": [14, 3, 6, 2, 6]})

# Print the dataframe
print('5'*100)
print(df)
print('6'*100)
print(df.mode())
import pandas as pd

# Creating the dataframe
df = pd.DataFrame({"A": [14, 4, 5, 4, 1],
                   "B": [5, 2, 54, 3, 2],
                   "C": [20, 20, 7, 3, 8],
                   "D": [14, 3, 6, 2, 6]})

# Print the dataframe
print('7'*100)
print(df)
print('8'*100)
print(df.mode(axis=1))
