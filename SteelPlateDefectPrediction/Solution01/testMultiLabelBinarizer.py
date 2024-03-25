# -*- coding: utf-8 -*-
# @Time : 2024/3/25 16:51
# @Author : nanji
# @Site : https://blog.csdn.net/NockinOnHeavensDoor/article/details/80234510
# @File : testMultiLabelBinarizer.py
# @Software: PyCharm 
# @Comment : 
from sklearn.preprocessing import MultiLabelBinarizer
mlb = MultiLabelBinarizer()

mlb.fit_transform([(1, 2), (3,)])
array([[1, 1, 0],
       [0, 0, 1]])
print(mlb.classes_)
