# -*- coding: utf-8 -*-
# @Time : 2024/4/15 10:28
# @Author : nanji
# @Site : https://blog.csdn.net/qq_24591139/article/details/100328640
# @File : testRandomUnderSampler.py
# @Software: PyCharm 
# @Comment : 
from imblearn.under_sampling import RandomUnderSampler
import  pandas as pd
import random as rd
import numpy as np
import math as ma

def typeicalSampling(group, typeicalFracDict):
    name = group.name
    frac = typeicalFracDict[name]
    return group.sample(frac=frac)

def group_sample(data_set,lable,typeicalFracDict):
    #分层抽样
    #data_set数据集
    #lable分层变量名
    #typeicalFracDict：分类抽样比例
    gbr=data_set.groupby(by=[lable])
    result=data_set.groupby(lable,group_keys=False).apply(typeicalSampling,typeicalFracDict)
    return result

data = pd.DataFrame({'id': [3566841, 6541227, 3512441, 3512441, 3512441,3512441, 3512441, 3512441, 3512441, 3512441],
                   'sex': ['male', 'Female', 'Female','male', 'Female', 'Female','male', 'Female','male', 'Female'],
                   'level': ['high', 'low', 'middle','high', 'low', 'middle','high', 'low', 'middle','middle']})

data_set=data
label='sex'
typicalFracDict = {
    'male': 0.8,
    'Female': 0.2
}
result=group_sample(data_set,label,typicalFracDict)
print(result)


cc = RandomUnderSampler(sampling_strategy={0: 50, 2: 100, 1: 100}, \
						random_state=0)
X_resampled,y_resampled=cc.fit_resample(X,y)
