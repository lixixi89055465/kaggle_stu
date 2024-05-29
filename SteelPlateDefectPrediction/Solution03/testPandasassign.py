# -*- coding: utf-8 -*-
# @Time : 2024/5/29 11:07
# @Author : nanji
# @Site : https://blog.csdn.net/weixin_43837522/article/details/133929047
# @File : testPandasassign.py
# @Software: PyCharm 
# @Comment :
import pandas as pd

df = pd.DataFrame({
	'A': range(1, 5),
	'B': range(5, 9)
})
new_df = df.assign(C=df['A'] + df['B'])
print('1' * 100)
print(new_df)
new_df = df.assign(C=df['A'] + df['B'], D=df['A'] * df['B'])
print('2' * 100)
print(new_df)

