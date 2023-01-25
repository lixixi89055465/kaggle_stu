import numpy as np
import pandas as pd
import seaborn as sns

import matplotlib.pyplot as plt

'''
1.综合散点图 - JoinGrid() 
# 可拆分绘制的散点图 
plot_joint() + ax_marg_x.hist() + ax_marg_y.hist() 
'''
sns.set_style('white')
# 设置风格

# tips=sns.load_dataset('tips')
tips = pd.read_csv('tips.csv')
print(tips.head())
# 导入数据

# 创建密度图
g = sns.JointGrid(x='total_bill', y='tip', data=tips)
# 创建一个绘图表格区域，设置好x,y对应数据


g.plot_joint(plt.scatter, color='m', edgecolor='white')  # 设置框内图表 ，scatter
g.ax_marg_x.hist(tips['total_bill'], color='b', alpha=.6,
                 bins=np.arange(0, 60, 3))  # 设置x轴直方图，注意bins是数组
g.ax_marg_y.hist(tips['tip'], color='r', alpha=.6,
                 orientation='horizontal',
                 bins=np.arange(0, 12, 1))  # 设置y轴直方图，注意需要orientation参数

from scipy import stats

# g.annotate(stats.pearsonr)
# g.annotate(stats.pearsonr)
# 设置 标注，可以为 pearsonr,spearmanr
plt.grid(linestyle='--')
