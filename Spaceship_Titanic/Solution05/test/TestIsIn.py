import pandas as pd

df = pd.DataFrame({'A': [1, 2, 3],
                   'B': ['a', 'b', 'c']})
print(df.isin([1, 3, 12, 'a']))
print('0' * 100)
print(df.isin({
    'A': [1, 3],
    'B': ['a', 7, 12]
}))
print('1' * 100)
print(df.isin({
    'A': [1, 3],
    'C': [4, 7, 12]
}))
df = pd.DataFrame({

    'A': [1, 2, 3],

    'B': ['a', 'b', 'f']

})

other = pd.DataFrame({
    'A': [1, 3, 3, 2],
    'B': ['e', 'f', 'f', 'e']
})
print('2'*100)
print(df.isin(other))
