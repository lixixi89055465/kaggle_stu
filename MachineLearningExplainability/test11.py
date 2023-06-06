import pandas as pd

data = {
    'brand': ['Python数据之道', '价值前瞻', '菜鸟数据之道', 'Python', 'Java'],
    'A': [10, 2, 5, 20, 16],
    'B': [4, 6, 8, 12, 10],
    'C': [8, 12, 18, 8, 2],
    'D': [6, 18, 14, 6, 12],
    'till years': [4, 1, 1, 30, 30]
}

df = pd.DataFrame(data=data)
print(df.columns)

a=list(df['A'].values)
b=df['B'].values
print(a)
a.pop()