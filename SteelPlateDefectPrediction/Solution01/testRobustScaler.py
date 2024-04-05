# -*- coding: utf-8 -*-
# @Time : 2024/4/4 16:12
# @Author : nanji
# @Site : https://blog.csdn.net/shenliang1985/article/details/112525250
# @File : testRobustScaler.py
# @Software: PyCharm 
# @Comment : 
import pandas as pd
from sklearn.preprocessing import RobustScaler

data = pd.DataFrame(
	{
		'a': [1, 2, 3],
		'b': [5, 6, 6],
		'c': [9, 100, 2]
	}
)
print(data.values)

robustlizer = RobustScaler(quantile_range=(25.0, 75.0))
robustlizer_data = robustlizer.fit_transform(data)
print(robustlizer_data)