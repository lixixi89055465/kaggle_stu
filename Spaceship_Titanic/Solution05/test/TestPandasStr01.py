import pandas as pd
import numpy as np

s = pd.Series(['A', 'B', 'C', 'Aaba', 'Baca', np.nan, 'CABA', 'dog', 'cat'])
print(s.str.lower())
print('0' * 100)
print(s.str.upper())
print('1' * 100)
print(s.str.len())
print('2' * 100)
idx = pd.Index([' jack', 'jill', 'jesse', 'frank'])
print(idx.str.strip())
print('3' * 100)
print(idx.str.lstrip())
print('4' * 100)
print(idx.str.rstrip())
print('5' * 100)
df = pd.DataFrame(np.random.randn(3, 2),
                  columns=[' Column A ', ' Column B '], index=range(3))

print(df)
print('6' * 100)
print(df.columns.str.strip())
print('7' * 100)
print(df.columns.str.lower())

print('8' * 100)
df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
print(df)
print('9' * 100)
s2 = pd.Series(['a_b_c', 'c_d_e', np.nan, 'f_g_h'])
print(s2.str.split('_'))
print(s2.str.split('_').str.get(1))
print('0' * 100)
print(print(s2.str.split('_').str[2]))
print('1' * 100)
print(s2.str.split('_', expand=True))
print(s2.str.split('_', expand=True)[1])
print('2' * 100)
print(s2.str.split('_', expand=True, n=1))

# print('3' * 100)
# print(s2.rsplit('_', expand=True, n=1))
print('4' * 100)
s3 = pd.Series(['A', 'B', 'C', 'Aaba', 'Baca', '', np.nan, 'CABA', 'dog', 'cat'])
print(s3)
s = pd.Series(['a', 'b', 'c', 'd'])
print(s.str.cat(sep=','))
print('5' * 100)
print(s.str.cat())
print('6' * 100)
t = pd.Series(['a', 'b', np.nan, 'd'])
print(t.str.cat(sep=','))

print('7' * 100)
a = pd.Series(['a', 'b', 'c']).str.cat(['A', 'B', 'C'], sep=',')
print(a)
print('8' * 100)

print(pd.Series(['a', 'b', 'c']).str.cat(sep=','))

print('9'*100)
print(pd.Series(['a', 'b','c']).str.cat(['x', 'y', '2'], sep=','))
print('0'*100)
import numpy,pandas
s=pandas.Series(['a_b_c','c_d_e',numpy.nan,'f_g_h'])
print(s.str.split('_'))

print('0'*100)
print(s.str.split('_'))
print('1'*100)
# print(s.str.split('_', -1))
print(s)
print(s.str.get(0))
print(s.str.get(1))
print(s.str.get(2))

print('2'*100)
print(s.str.join('!'))
print('3'*100)
print(s.str.join('?'))

print('4'*100)
print(s.str.contains('d'))
print(s.str.repeat(3))
print('5'*100)
print(s.str.pad(10, fillchar='?'))
print(s.str.pad(10, side='right', fillchar='?'))
print('6'*100)
print(s.str.center(10, fillchar='?'))
print(s.str.ljust(10, fillchar='?'))
print('7'*100)
print(s.str.rjust(10, fillchar='?'))
print('8'*100)
print(s.str.zfill(10))
print('9'*100)
print(s.str.wrap(3))