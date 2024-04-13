# -*- coding: utf-8 -*-
# @Time    : 2024/4/12 下午11:54
# @Author  : nanji
# @Site    : 
# @File    : testPartial.py
# @Software: PyCharm 
# @Comment :
import functools
def sum(x,y):
	return x+y

# print(sum(5,6))
sumY=functools.partial(sum,5)
print('0'*100)
print(sumY(6))
