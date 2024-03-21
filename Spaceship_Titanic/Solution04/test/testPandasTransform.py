# -*- coding: utf-8 -*-
# @Time : 2024/3/21 16:24
# @Author : nanji
# @Site : 
# @File : testPandasTransform.py
# @Software: PyCharm
# @Comment : https://blog.csdn.net/weixin_43790276/article/details/126911525

# coding=utf-8
import pandas as pd
import numpy as np

df = pd.DataFrame({'Col-1': [1, 3, 5], 'Col-2': [5, 7, 9]})
print(df)
res1 = df.transform(np.square)
print(res1)
res2 = df.transform('sqrt')
print(res2)
res3 = df.transform(lambda x: x*10)
print(res3)


