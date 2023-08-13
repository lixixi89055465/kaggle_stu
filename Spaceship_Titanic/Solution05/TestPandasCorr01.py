import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas import DataFrame
import pandas as pd

x = [a for a in range(100)]


# 构造一元二次方程，非线性关系
def y_x(x):
    return 2 * x ** 2 + 4


y = [y_x(i) for i in x]

data = DataFrame({'x': x, 'y': y})

# 查看下data的数据结构
print('0'*100)
print(data.head())
print('1'*100)
print(data.corr())

print('2'*100)
print(data.corr(method='spearman'))
print('3'*100)

print(data.corr(method='kendall'))
print('4'*100)
print(data.corr()['x'])
print('5'*100)
print(data['x'].corr(data['y'],method='pearson' ))
print(data['x'].corr(data['y'],method='spearman' ))
print(data['x'].corr(data['y'],method='kendall' ))
#pair plots of entire dataset
pp = sns.pairplot(data1, hue = 'Transported', palette = 'deep', size=1.2, diag_kind = 'kde', diag_kws=dict(shade=True), plot_kws=dict(s=10) )
pp.set(xticklabels=[])