# -*- coding: utf-8 -*-
# @Time : 2024/4/17 10:50
# @Author : nanji
# @Site : 
# @File : testpinv.py
# @Software: PyCharm 
# @Comment :
from numpy import array
from numpy.linalg import pinv
from scipy.linalg import svd

# # define a matrix
# A = array([[1, 2], [3, 4], [5, 6]])
# print(A)
# # SVD
# U, s, VT = svd(A)
# print(U)
# print(s)
# print(VT)
# print('0' * 100)
# A = array([
# 	[0.1, 0.2],
# 	[0.3, 0.4],
# 	[0.5, 0.6],
# 	[0.7, 0.8],
# ])
# print(A)
# B=pinv(A)
# print('0'*100)
# print(B)
print('2' * 100)

from numpy import zeros, diag

# A = array([
# 	[1, 2],
# 	[3, 4],
# 	[5, 6],
# ])
# print(A)
# U, s, VT = svd(A)
# Sigma = zeros((A.shape[0], A.shape[1]))
#
# Sigma[:A.shape[1], :A.shape[1]] = diag(s)
# B=U.dot(Sigma.dot(VT))
# print('0'*100)
# print(B)

from numpy import array
from numpy.linalg import pinv

A = array([
	[0.1, 0.2],
	[0.3, 0.4],
	[0.5, 0.6],
	[0.7, 0.8],
])
print('3'*100)
print(A)
B = pinv(A)
print(B)
