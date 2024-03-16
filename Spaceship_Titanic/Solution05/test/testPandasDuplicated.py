# -*- coding: utf-8 -*-
# @Time : 2024/3/2 13:03
# @Author : nanji
# @Site : 
# @File : testPandasDuplicated.py
# @Software: PyCharm 
# @Comment :https://blog.csdn.net/liuweiyuxiang/article/details/90940160


import pandas as pd
df= pd.DataFrame({'k1': [ 's1']* 3 + ['s2']* 5,'k2' : [1, 1, 2, 3, 3, 4, 4,4]})
print(df)
result1=df.duplicated()
result2=df.duplicated(keep='last')
result3=df.duplicated(keep=False)
result4=df.duplicated('k1')
result5=df.duplicated(['k1','k2'])
