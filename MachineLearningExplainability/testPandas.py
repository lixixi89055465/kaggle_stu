import pandas as pd

print(f'pandas version: {pd.__version__}')

# pandas version: 1.3.0rc1
data = {
    'brand': ['Python数据之道', '价值前瞻', '菜鸟数据之道', 'Python', 'Java'],
    'A': [10, 2, 5, 20, 16],
    'B': [4, 6, 8, 12, 10],
    'C': [8, 12, 18, 8, 2],
    'D': [6, 18, 14, 6, 12],
    'till years': [4, 1, 1, 30, 30]
}

df = pd.DataFrame(data=data)
print(df)
print('0' * 100)
print(df.query(('brand == "Python数据之道"')))
print('1' * 100)

# 通过变量来筛选数据，在变量前使用 @ 符号即可

name = 'Python数据之道'

print(df.query('brand == @name'))
print('2' * 100)
print(df.query('brand in ["Python数据之道","价值前瞻"]'))
print('3' * 100)
print(df.query('`till years` < 5'))
print('4' * 100)
print(df[(df['brand'] == "Python数据之道") & (df['A'] > 2) & (df['C'] > 4)])

