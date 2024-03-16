# -*- coding: utf-8 -*-
# @Time : 2024/3/14 10:54
# @Author : nanji
# @Site : 
# @File : testBinary.py
# @Software: PyCharm 
# @Comment : https://hg95.github.io/sklearn-notes/Chapter3/preprocessing.Binarizer%E7%94%A8%E6%B3%95.html
# from sklearn.preprocessing import Binarizer
# X = [[ 1., -1.,  2.],
#      [ 2.,  0.,  0.],
#      [ 0.,  1., -1.]]
# transformer = Binarizer().fit(X)  # fit does nothing.
# a=transformer.transform(X)
# print(a)


from  sklearn.preprocessing import  Binarizer
from  sklearn import preprocessing

X = [[ 1., -1.,  2.],
      [ 2.,  0.,  0.],
      [ 0.,  1., -1.]]

transform = Binarizer(threshold=0.0)
newX=transform.fit_transform(X)
# print(mm)

# transform = Binarizer(threshold=0.0).fit(X)
# newX = transform.transform(X)

binarizer = preprocessing.Binarizer().fit(X)  # fit does nothing
print(binarizer)
#Binarizer(copy=True, threshold=0.0)

print(binarizer.transform(X))
'''
[[1. 0. 1.]
 [1. 0. 0.]
 [0. 1. 0.]]
'''

binarizer = preprocessing.Binarizer(threshold=1.1)
print(binarizer.transform(X))
'''
[[0. 0. 1.]
 [1. 0. 0.]
 [0. 0. 0.]]
'''

