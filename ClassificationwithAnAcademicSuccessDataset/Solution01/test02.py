# -*- coding: utf-8 -*-
# @Time    : 2024/6/10 上午10:22
# @Author  : nanji
# @Site    : 
# @File    : test02.py
# @Software: PyCharm 
# @Comment :
import pandas as pd

from sklearn.preprocessing import LabelEncoder
labelEncoder=LabelEncoder()
Target='Target'
p1=pd.read_csv('../input/playground-series-s4e6/train.csv')
print(p1.columns)
labelEncoder.fit(p1[Target])
test1=pd.read_csv('./submission05.csv')
test1[Target]=labelEncoder.inverse_transform(test1[Target])
print(test1.head())
test1.to_csv('submission07.csv',index=False)
