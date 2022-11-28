import pandas as pd
import numpy as np

df = pd.DataFrame({'城市': ['北京', '广州', '天津', '上海', '杭州', '成都', '澳门', '南京'],

                   '收入': [10000, 10000, 5000, 5002, 40000, 50000, 8000, 5000],

                   '年龄': [50, 43, 34, 40, 25, 25, 45, 32]})
df.set_index([["一", "二", "三", "四", "五", "六", "七", "八"]], inplace=True)
print(df)
# print(df.loc['一'])
# print(df.loc['一': '二'])
# print(df.loc[['一', '二']])
# print(df.loc['一': '四'])
# print(df.loc['一', '城市'])
# print(df.loc['一':'二', '城市':'收入'])
# print(df.loc[:,'城市'])
# print(df.loc[0])
print(df.iloc[0])
print(df.iloc[1])
print(df.iloc[0:1])

print(df.iloc[0:2])
print('1'*100)
print(df.iloc[0:8:2])

print(df.iloc[:, 1:3])

print(df.iloc[:, [1, 2]])
print(df.iloc[[0, 1], [0, 1]])