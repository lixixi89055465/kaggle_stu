import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style('darkgrid')
sns.set_context('paper')
# 设置风格、尺度
import warnings

warnings.filterwarnings('ignore')
# 不发出警告
'''
1。基本设置 
绘制直方图 
'''
tips = pd.read_csv('tips.csv')
tips.head()
# 导入数据

g = sns.FacetGrid(tips, col='time', row='smoker')
# 创建一个绘图表格区域，设置好row,col 并分组

g.map(plt.hist, 'total_bill', alpha=0.5, color='k', bins=10)
plt.show()
'''
1.基本设置 
绘制直方图 
'''
g = sns.FacetGrid(tips,
                  col='day',
                  size=4,  # 图表大小
                  aspect=1,  # 图表长宽比
                  )
g.map(
    plt.hist, 'total_bill',
    alpha=0.5,
    bins=10,
    histtype='step',  # bar , barstacked, step, stepfilled
    color='k',
)

plt.show()

'''
1.基本设置
绘制散点图
'''
g = sns.FacetGrid(tips, col='time', row='smoker')
# 创建一个绘图表格区域，设置好row,col并分组
g.map(plt.scatter,
      'total_bill', 'tip',  # share{x,y}-> 设置x,y数据
      edgecolor='w', s=40, linewidth=1)  # 设置点大小，描边宽度及颜色
g.add_legend()
# 添加图例

plt.show()

# 图表矩阵
# 分类
g = sns.FacetGrid(tips, col='time', hue='smoker')
# 创建一个绘图表格区域，设置好col并分组，按照hue分类
g.map(plt.scatter,
      "total_bill", "tip",  # share {x,y} - > 设置x,y数据
      edgecolor='w', s=40, linewidth=1)  # 设置点大小，描边宽度及颜色
g.add_legend()
# 添加图例
plt.show()

# 图表矩阵
attend = pd.read_csv('attention.csv')
# attend[['solutions', 'score']] = attend.apply(lambda x: (str(x[0]), str(x[1])), axis=1, result_type="expand")
# attend.apply(lambda x: print(x))
print(attend.head())
print(attend.columns)
# attend['solutions'] = attend['solutions'].map(lambda x: int(x))
# attend['score'] = attend['score'].map(lambda x: float(x))
# 加载数据
g = sns.FacetGrid(attend, col='subject', col_wrap=5,  # 设置每行的图表数量
                  size=1.5)
g.map(plt.plot, 'solutions', 'score',
      marker='o', color='gray', linewidth=2
      )
# 绘制图表矩阵
# g.set(
#     xlim=(0, 4),
#     ylim=(0, 10),
#     xticks=[0, 1, 2, 3, 4],
#     yticks=[0, 2, 4, 6, 8, 10]
# )
# 设置x,y轴刻度
plt.show()
print("1" * 100)
print(attend[attend['subject'] == 1])
