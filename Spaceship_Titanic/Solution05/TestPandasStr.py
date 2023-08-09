import pandas as pd

d = {'gene': {'a': 'gene1', 'b': 'gene2', 'c': 'gene3', 'd': 'gene4'},
     'expression': {'a': 'low:0', 'b': 'mid:3', 'c': 'mid:4', 'd': 'high:9'},
     'description': {'a': 'transposon element', 'b': 'nuclear genes', 'c': 'retrotransposon', 'd': 'unknown'}}
df = pd.DataFrame(d)
print(df)
df1 = df[df['description'].str.contains('transposon')]
print('0' * 100)
print(df1)
df2 = df[df['description'].str.contains('transposon')]
print(df2)
print('1' * 100)
df1 = df['expression'].str.split(':', expand=False)
print(df1)
df[['exp1', 'exp2']] = df['expression'].str.split(':', expand=True)
print(df)
print('2' * 100)
print(df['exp1'].dtype)
print('3' * 100)
print(df['exp2'].dtype)
print('4' * 100)
df['exp2'] = df['exp2'].astype(int)
print('5' * 100)
print(df['exp2'].dtype)
print('6' * 100)
df['exp2'] = df['exp2'].astype(int)
print('7' * 100)
print(df['exp2'].dtype)
df = pd.DataFrame(d)
print('8' * 100)
df['gene'] = df['gene'].str.replace('gene', 'Gene')
print(df)
df1 = df[df['expression'].str.startswith('m')]
print('9' * 100)
print(df1)

s = df['expression'].str.findall('[a-z]+')
print('0' * 100)
print(s)

print('1'*100)

df1['expression']=df1['expression'].str.lstrip('mid:')
print(df1)
