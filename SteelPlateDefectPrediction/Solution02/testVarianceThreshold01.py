# -*- coding: utf-8 -*-
# @Time : 2024/5/1 17:06
# @Author : nanji
# @Site :https://zhuanlan.zhihu.com/p/141506312
# @File : testVarianceThreshold01.py
# @Software: PyCharm 
# @Comment :
from sklearn.feature_selection import VarianceThreshold

# X = [[0, 0, 1], [0, 1, 0], [1, 0, 0], [0, 1, 1], [0, 1, 0], [0, 1, 1]]
# sel = VarianceThreshold(threshold=(.8 * (1 - .8)))
# X_new=sel.fit_transform(X)
# print(X_new)

X=[
	[100, 1, 2, 3],
	[100, 4, 5, 6],
	[100, 7, 8, 9],
	[101, 11, 12, 13]
]
sel=VarianceThreshold(1)
sel.fit(X)   #获得方差，不需要y
print('Variances is %s'%sel.variances_)
print('After transform is \n%s'%sel.transform(X))
print('The surport is %s'%sel.get_support(True))#如果为True那么返回的是被选中的特征的下标
print('The surport is %s'%sel.get_support(False))#如果为FALSE那么返回的是布尔类型的列表，反应是否选中这列特征

