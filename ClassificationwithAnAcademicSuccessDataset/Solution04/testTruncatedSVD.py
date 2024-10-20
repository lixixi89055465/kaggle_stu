# -*- coding: utf-8 -*-
# @Time : 2024/10/20 16:28
# @Author : nanji
# @Site : https://blog.csdn.net/qq_32172681/article/details/99191092
# @File : testTruncatedSVD.py
# @Software: PyCharm 
# @Comment :
from sklearn.decomposition import TruncatedSVD

from sklearn.datasets import load_iris

svd = TruncatedSVD(2)

iris_data = load_iris()['data']
print('0' * 100)
print(iris_data[:5])
iris_transformed = svd.fit_transform(iris_data)
print('1' * 100)
print(iris_transformed[:5])
