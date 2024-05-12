# -*- coding: utf-8 -*-
# @Time    : 2024/5/12 下午11:01
# @Author  : nanji
# @Site    :https://blog.csdn.net/weixin_43746433/article/details/97808078
# @File    : testNuSVC.py
# @Software: PyCharm 
# @Comment :
import numpy as np

X = np.array([[-1, -1], [-2, -1], [1, 1], [2, 1]])
y = np.array([1, 1, 2, 2])
from sklearn.svm import NuSVC

# clf = NuSVC()
# r1 = clf.fit(X, y)
# print(r1)
# r2 = clf.predict([[-0.8, -1]])
# print('0' * 100)
#
# print(r2)

from sklearn.datasets import fetch_lfw_people

# faces = fetch_lfw_people(min_faces_per_person=60)
# print(faces.target_names)
# print(faces.images.shape)

from sklearn.svm import LinearSVC

X = [[0], [1], [2], [3]]
Y = [0, 1, 2, 3]
clf = LinearSVC(decision_function_shape='ovo')  # ovo为一对一
r1 = clf.fit(X, Y)
print(r1)
