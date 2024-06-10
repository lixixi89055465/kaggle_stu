# -*- coding: utf-8 -*-
# @Time    : 2024/6/9 下午9:48
# @Author  : nanji
# @Site    : https://blog.csdn.net/xiezhen_zheng/article/details/106097276
# @File    : testiterrows.py
# @Software: PyCharm 
# @Comment :
import pandas as pd

df = pd.DataFrame(
	[
		('E146', 100.92, '[-inf ~ -999998.0]'),
		('E138', 107.92, '[-999998.0 ~ 2]'),
		('E095', 116.92, '[1.5 ~ 3.5]')],
	columns=['name', 'score', 'value'])
print(df)

print('0'*100)
for row_index, row in df.iterrows():
	print('行号：', row_index)
	print('第{} 行的值：'.format(row_index))
	print(row)
	print('第{} 行 value 列 的值：'.format(row_index),row['value'])
