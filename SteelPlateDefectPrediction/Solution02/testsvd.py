# -*- coding: utf-8 -*-
# @Time : 2024/4/17 11:28
# @Author : nanji
# @Site : https://zhuanlan.zhihu.com/p/134512367
# @File : testsvd.py
# @Software: PyCharm 
# @Comment :
from numpy import array, zeros, diag
from scipy.linalg import svd

A = array([
	[i for i in range(1, 11)],
	[i for i in range(11, 21)],
	[i for i in range(21, 31)],
])
print(A)

U, s, VT = svd(A)
Sigma = zeros((A.shape[0], A.shape[1]))
Sigma[:A.shape[0], :A.shape[0]] = diag(s)
n_elements = 2
Sigma = Sigma[:, :n_elements]
VT = VT[:n_elements, :]
B = U.dot(Sigma.dot(VT))
print(B)
T = U.dot(Sigma)
print(T)
T = A.dot(VT.T)
print(T)
