import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

'''
课程2.5 分布数据可视化 - 散点图 
joinplot() / pairplot()

1.综合散点图 - joinplot() 
散点图 + 分布图 
'''
tips=pd.read_csv('tips.csv')

# 创建数据

print(tips.head())

sns.pointplot(x='time', y='total_bill', hue='smoker',data=tips,
              palette='hls',
              dodge=True,  # 设置点是否分开
              join=True,  # 是否连线
              markers=['o', 'x'],
              linestyles=['-', '--'],  # 设置点样式、线形

              )
# tips.groupby(['time', 'smoker']).mean()['total_bill']
# 计算数据
# 用法和barplot相似
