import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

rs = np.random.RandomState(2)  # 设定随机数种子
df = pd.DataFrame(rs.randn(100, 2),
                  columns=['A', 'B'])

sns.kdeplot(
    x=df['A'],
    y=df['B'],
    cbar=True,  # 是否显示颜色图例
    shade=True,  # 是否填充
    cmap='Reds_r',  # 设置调色盘
    shade_lowest=False,  # 最外围颜色是否显示
    n_levels=100  # 曲线个数（如果非常多，则会越平滑 )
)
# 两个维度数据生成曲线密度图 ，以颜色作为密度衰减显示
plt.grid(linestyle='--')
plt.scatter(df['A'], df['B'], s=5, alpha=.5, color='k')
sns.rugplot(df['A'], color='g', axis='x', alpha=.5)
sns.rugplot(df['B'], color='r', axis='y', alpha=.5)
# 注意设置x,y轴

plt.show()
'''
2.密度图-kdeplot() 
两个样本数据密度分布图 
多个密度图 
'''
rs1 = np.random.RandomState(2)
rs2 = np.random.RandomState(5)
df1 = pd.DataFrame(rs1.randn(100, 2) + 2, columns=['A', 'B'])
df2 = pd.DataFrame(rs2.randn(100, 2) - 2, columns=['A', 'B'])

# 创建数据
sns.kdeplot(x=df1['A'], y=df1['B'], cmap='Greens', shade=True, shade_lowest=False)
sns.kdeplot(x=df2['A'], y=df2['B'], cmap='Blues', shade=True, shade_lowest=False,n_levels=100)
plt.show()
