# -*- coding: utf-8 -*-
# @Time : 2024/4/17 13:53
# @Author : nanji
# @Site : https://zhuanlan.zhihu.com/p/134512367
# @File : testTruncatedSVD.py
# @Software: PyCharm 
# @Comment :
from numpy import array
from sklearn.decomposition import TruncatedSVD

A = array([
	[i for i in range(1, 11)],
	[i for i in range(11, 21)],
	[i for i in range(21, 31)],
])
print(A)
svd = TruncatedSVD(n_components=2)
svd.fit(A)
r1 = svd.transform(A)
print(r1)
