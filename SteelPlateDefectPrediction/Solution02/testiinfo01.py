# -*- coding: utf-8 -*-
# @Time : 2024/5/23 13:38
# @Author : nanji
# @Site : https://blog.csdn.net/weixin_44025103/article/details/125660180
# @File : testiinfo01.py
# @Software: PyCharm 
# @Comment : 
import warnings

warnings.filterwarnings('ignore')
import numpy as np

int8 = np.iinfo(np.int8)
int16 = np.iinfo(np.int16)
int32 = np.iinfo(np.int32)
int64 = np.iinfo(np.int64)

print(int8)
print(int16)
print(int32)
print(int64)
print('9'*100)
import numpy as np
float16 = np.finfo(np.float16)
float32 = np.finfo(np.float32)
float64 = np.finfo(np.float64)

print(float16)
print(float32)
print(float64)



