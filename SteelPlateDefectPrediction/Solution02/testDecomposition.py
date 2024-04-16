# -*- coding: utf-8 -*-
# @Time : 2024/4/16 22:13
# @Author : nanji
# @Site : https://zhuanlan.zhihu.com/p/134512367
# @File : testDecomposition.py
# @Software: PyCharm 
# @Comment : 
from numpy import array
from scipy.linalg import svd

A = array([
	[1, 2], [3, 4], [5, 6]
])
print(A)
U,s,VT=svd(A)
print(U)
print(s)
print(VT)