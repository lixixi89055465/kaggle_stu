# Create a simple dataframe

# importing pandas as pd
import pandas as pd

import pandas as pd

df = pd.DataFrame({'A': {0: 'a', 1: 'b', 2: 'c'},
                   'B': {0: 1, 1: 3, 2: 5},
                   'C': {0: 2, 1: 4, 2: 6}})
print(df)
print('1' * 100)
df = pd.melt(df, id_vars='A', value_vars='C')
print(df)

print('2' * 100)
df = pd.melt(df, id_vars='A')
print(df)
