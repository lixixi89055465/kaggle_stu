import pandas as pd
import numpy as np

codes, uniques = pd.factorize(['b', None, 'a', 'c', 'b'])
print(codes)
print(uniques)

cat = pd.Categorical(['a', 'b', 'c', 'a'], categories=['a', 'b', 'c'])
print(cat)

codes, uniques = pd.factorize(cat)
print(codes)
print(uniques)

print('1' * 50)
cat = pd.Series(['a', 'a', 'c'])
codes, uniques = pd.factorize(cat)
print(codes)
print(uniques)
print('2' * 50)

values=pd.array([1,2,1,np.nan])
codes,uniques=pd.factorize(values)
print(codes)
print(uniques)

print('3' * 50)
code,uniques=pd.factorize(values,na_sentinel=None)
print(codes)
print(uniques)
