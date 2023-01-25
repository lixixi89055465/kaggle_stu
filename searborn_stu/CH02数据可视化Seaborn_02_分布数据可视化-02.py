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
rs = np.random.RandomState(2)
df = pd.DataFrame(rs.randn(200, 2), columns=['A', 'B'])
# 创建数据
sns.jointplot(
    x=df['A'],
    y=df['B'],  # 设置 XY 轴，显示columns 名称
    data=df,  # 设置数据
    color='k',  # 设置颜色
    s=50,  #
    edgecolor='w', linewidth=1,  # 设置散点大小，边缘线颜色及宽度（只针对 scatter)
    kind='scatter',  # 设置类型 ： 'scatter','reg','resid','kde','hex'
    space=.2,  # 设置散点图和布局图的间距
    size=8,  # 图标大小
    ratio=5,  # 散点图与布局图高度比，整形
    marginal_kws=dict(bins=15, rug=True)  # 设置柱状图箱数 ，是否设置rug

)
plt.show()
