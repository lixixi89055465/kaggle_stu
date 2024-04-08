# -*- coding: utf-8 -*-
# @Time : 2024/4/8 9:49
# @Author : nanji
# @Site : https://blog.csdn.net/weixin_48249563/article/details/114899818
# @File : testmissingno.py
# @Software: PyCharm 
# @Comment : 

import missingno as msno
from sklearn.datasets import load_boston
import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')

data, target = load_boston()['data'], load_boston()['target']
df = pd.concat([pd.DataFrame(data, columns=[str('features' + str(i)) for i in range(13)]),
				pd.DataFrame(target, columns=['target'])], axis=1)
print(df.head(10))

print('0' * 100)
df1 = df.where(df != 0, np.nan)
print(df1.isnull().sum())
print('1' * 100)
msno.matrix(df1, labels=True)

print('2' * 100)
msno.matrix(df1, labels=True)  # 无效数据密度显示
plt.show()
