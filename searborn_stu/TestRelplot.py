import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# input = pd.read_csv('tips.csv')
#
# print(input.head(5))
# print(input.shape)
# sns.relplot(x='total_bill', y='tip', input=input)
#
# sns.relplot(x='total_bill', y='tip', hue='smoker', input=input)
#
# sns.relplot(x='total_bill', y='tip', hue='smoker', style='sex', input=input)

# size 可以转化为float
# sns.relplot(x='total_bill', y='tip', hue='size', input=input)
# 5为信息
# sns.relplot(x='total_bill',
#             y='tip',
#             hue='smoker',
#             style='sex',
#             size='size',
#             sizes=(15, 200),
#             input=input
#
#             )
# plt.show()
# input = pd.DataFrame({'time': np.arange(500),
#                      'value': np.random.randn(500).cumsum()})
# print(input.head())
#
# sns.relplot(x='time', y='value', kind='line', input=input)
# 用线强调连续性
# plt.show()
#
# print("2" * 100)
# fmri = pd.read_csv('fmri.csv')
# print(fmri.head(5))
# print(fmri.shape)
# his = {}
# for input in fmri['timepoint'].unique():
#     his[str(input)] = fmri[fmri['timepoint'] == input]
# print(his)
# 默认行为：x是通过绘制平均值周围的平均值和95%置信区间来聚合每个值的多少测量
# 绘制：置信区间来表示每个时间点的分布范围
# sns.relplot(x='timepoint', y='signal', kind='line', input=fmri)
# sns.relplot(x='timepoint', y='signal', kind='line', ci=None,
#             input=fmri)

# 绘制标准差来表示每个时间点的分布范围
# sns.relplot(
#     x='timepoint',
#     y='signal',
#     kind='line',
#     # ci='sd',
# errorbar='sd',
#     input=fmri
#
# )

# sns.relplot(x='timepoint', y='signal', input=fmri)
# 用语义映射绘制数据子集
# sns.relplot(
#     x='timepoint',
#     y='signal',
#     kind='line',
#     hue='event',
#     input=fmri
# )
# sns.relplot(x='timepoint', y='signal',
#             hue='event', style='region',
#             kind='line', input=fmri
#             )
print('3' * 100)
# print(fmri.event.unique())
# print(fmri.query("event=='stim'"))
# dot = pd.read_csv('dots.csv')
# print(dot.head(5))

# sns.relplot(x='time', y='firing_rate', hue='coherence',
#             style='choice', kind='line', input=dot)

# sns.relplot(x='time',y='firing_rate',size='coherence',style='choice',
#             kind='line',input=dot)
# print('4'*100)
# sns.relplot(x='total_bill',y='tip',hue='smoker',
#             col='time',input=input
#             )
# 显示与facet的多个关系
print('5' * 100)
data = pd.read_csv('fmri.csv')
print(data.head(5))
print(data.shape)
# sns.relplot(
#     x='timepoint', y='signal', hue='subject',
#     col='region', row='event', height=3,
#     kind='line', input=input
# )
sns.relplot(x='timepoint', y='signal',
            hue='event', style='event',
            col='subject', col_wrap=5, height=3, kind='line',
            data=data
            )
plt.show()
