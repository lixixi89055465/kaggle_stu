# -*- coding: utf-8 -*-
# @Time : 2024/4/16 22:28
# @Author : nanji
# @Site : https://zhuanlan.zhihu.com/p/134512367
# @File : testSVD01.py
# @Software: PyCharm 
# @Comment :

from numpy import array
from scipy.linalg import  svd
# A=array([[1,2],[3,4],[5,6]])
# A = array([[1, 2], [3, 4], [5, 6]])
# print(A)
# # SVD
# U, s, VT = svd(A)
# print(U)
# print(s)
# print(VT)

from sklearn.decomposition import TruncatedSVD
print('0'*100)
A=array([
	[i for i in range(1,11)],
	[i for i in range(11,21)],
	[i for i in range(21,31)],
])
print(A)
#svd
svd=TruncatedSVD(n_components=2)
svd.fit(A)
r1=svd.fit_transform(A)
print(r1)