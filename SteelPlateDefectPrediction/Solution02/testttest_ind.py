# -*- coding: utf-8 -*-
# @Time : 2024/4/13 17:13
# @Author : nanji
# @Site : https://blog.csdn.net/qq_37006625/article/details/127937428
# @File : testttest_ind.py
# @Software: PyCharm 
# @Comment : 

import sys
import scipy.stats  as stats
A=stats.norm.rvs(loc=1,scale=1,size=(100))
B=stats.norm.rvs(loc=3,scale=1,size=(100))
r1=stats.levene(A,B)
print(r1)
print('0'*100)
r2=stats.ttest_ind(A,B,equal_var=False)
print(r2)





