import numpy as np
import pandas as pd
import seaborn as sns

import matplotlib.pyplot as plt

sns.set_style('whitegrid')
sns.set_context('paper')
# 设置风格、尺度
import warnings
warnings.filterwarnings('ignore')
#不发出警告

'''
1.stripplot() 
按照不同类别对样本数据进行分布散点图绘制 

'''
tips= pd.read_csv('tips.csv')
#加载数据
tips.head()
print(tips['day'].value_counts())

