import numpy as np
from math import log
import operator
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = [u'SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False
# 修改全局配置
font_options = {
    'family' : 'serif', # 设置字体家族
    'serif' : 'simsun', # 设置字体
}
plt.rc('font',**font_options)
plt.plot(list(np.arange(10)))
plt.xlabel('这是X轴呀')
plt.ylabel('这是Y轴呀')
plt.title('标题')
plt.show()