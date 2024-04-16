# -*- coding: utf-8 -*-
# @Time : 2024/4/16 13:45
# @Author : nanji
# @Site : https://blog.csdn.net/qq_41185868/article/details/115712219
# @File : testPool.py
# @Software: PyCharm 
# @Comment : 
from catboost import Pool

train_datapool = Pool(data=[
	[12, 14, 16, 18],
	[23, 25, 27, 29],
	[32, 34, 36, 38]],
	label=[10, 20, 30],
	weight=[0.1, 0.2, 0.3]
)

