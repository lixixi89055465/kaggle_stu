# -*- coding: utf-8 -*-
# @Time : 2024/4/5 17:31
# @Author : nanji
# @Site : https://blog.csdn.net/brucewong0516/article/details/80226990
# @File : testnpwhere.py
# @Software: PyCharm 
# @Comment :
import numpy as np
import pandas as pd

s = pd.Series(range(5))
# print(s.where(s > 0))
# print(s.mask(s > 0))
# r1=s.where(s > 1, 10)   #cond = s > 1,other = 10
df = pd.DataFrame(np.arange(10).reshape(-1, 2), columns=['A', 'B'])
m = df % 3 == 0

print(df.where(m, -df))
