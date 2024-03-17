# -*- coding: utf-8 -*-
# @Time : 2024/3/17 11:26
# @Author : nanji
# @Site : 
# @File : testduplicated.py
# @Software: PyCharm 
# @Comment : https://blog.csdn.net/wq_ocean_/article/details/108986252

import pandas as pd

df = pd.DataFrame({
	'brand': ['YumYum', 'YumYum', 'YumYum', 'Indomie', 'Indomie', 'Indomie'],
	'style': ['cup', 'cup', 'cup', 'cup', 'pack', 'pack'],
	'rating': [4, 4, 4, 3.5, 15, 5]})
print(df)
print(df.duplicated())
a=df.duplicated(keep='last')
print('0'*100)
print(a)
b=df.duplicated(subset=['brand'])
print('1'*100)
print(b)
print('2'*100)
c=df.duplicated(keep=False)
print(c)
print('3'*100)
print(df.drop_duplicates())