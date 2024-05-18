# -*- coding: utf-8 -*-
# @Time : 2024/5/18 11:05
# @Author : nanji
# @Site :https://blog.csdn.net/qq_42363032/article/details/121204166
# @File : testkstest01.py
# @Software: PyCharm 
# @Comment :
from scipy import stats
import pandas as pd

# scipy包是一个高级的科学计算库，它和Numpy联系很密切，Scipy一般都是操控Numpy数组来进行科学计算

data = [87, 77, 92, 68, 80, 78, 84, 77, 81, 80, 80, 77, 92, 86,
		76, 80, 81, 75, 77, 72, 81, 72, 84, 86, 80, 68, 77, 87,
		76, 77, 78, 92, 75, 80, 78]
# 样本数据，35位健康男性在未进食之前的血糖浓度
df = pd.DataFrame(data, columns=['value'])
u = df['value'].mean()
std = df['value'].std()
r1=stats.kstest(df['value'], 'norm', (u, std))
print(r1)

