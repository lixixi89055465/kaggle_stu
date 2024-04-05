# -*- coding: utf-8 -*-
# @Time : 2024/4/4 14:00
# @Author : nanji
# @Site : https://blog.csdn.net/weixin_44903299/article/details/115326809
# @File : testselect_dtypes.py
# @Software: PyCharm 
# @Comment :

import pandas as pd

df = pd.DataFrame({
	'a': [1, 2] * 3,
	'b': [True, False] * 3,
	'c': [1.0, 2.0] * 3,
	'd': ['1', '2'] * 3
})
print(df.info())
r1=df.select_dtypes(include = ['int64','float64'])
print('0'*100)
print(r1)
r2=df.select_dtypes(include = 'int64')
print('1'*100)
print(r2)

r3=df.select_dtypes(include = 'object')
print('2'*100)
print(r3)
r4=df.select_dtypes(include = ['int64','float64'])
print('3'*100)
print(r4)
print('4'*100)
r5=df.select_dtypes(include = 'bool')
print(r5)
print('5'*100)
r6=df.select_dtypes(exclude = 'bool')
print(r6)
numerical_fea = df.select_dtypes(include = 'int64').columns

print(numerical_fea)

print('6'*100)
numerical_fea = list(df.select_dtypes(include =['int64','float64']).columns)
print(numerical_fea)