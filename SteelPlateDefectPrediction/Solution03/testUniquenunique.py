# -*- coding: utf-8 -*-
# @Time    : 2024/5/26 下午3:41
# @Author  : nanji
# @Site    : https://blog.csdn.net/yeshang_lady/article/details/105345653
# @File    : testUniquenunique.py
# @Software: PyCharm 
# @Comment :
import pandas as pd
import numpy as np

a = pd.Series([1, 3, 3, 4, 2, 1])
print("a中的不同值", list(a.unique()))
print('0' * 100)

print("a中的不同值的个数", a.nunique())

print('1' * 100)
a = pd.Series([1, 3, None, 3, 4, 2])
print('a中的不同值 ：', list(a.unique()))
print('a中的不同值(排除None）的个数 ：', a.unique())
print('a中的不同值 ：(包括None）的个数：', a.nunique(dropna=False))
