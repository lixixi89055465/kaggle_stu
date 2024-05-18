# -*- coding: utf-8 -*-
# @Time : 2024/5/18 10:31
# @Author : nanji
# @Site :https://blog.csdn.net/qq_36056219/article/details/112118602
# @File : teststats_norm01.py
# @Software: PyCharm 
# @Comment :
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 8))
x = np.linspace(-5, 5, num=20)
plt.subplot(2, 2, 1)
# 第一种调用方式
gauss1 = stats.norm(loc=0, scale=2)  # loc : mean均值，scale:standard deviation 标准差
gauss2 = stats.norm(loc=1, scale=3)
y1 = gauss1.pdf(x)
y2 = gauss2.pdf(x)

plt.plot(x, y1, color='orange', label='u=0,sigma=2')
plt.plot(x, y2, color='green', label='u=0,sigma=2')
plt.legend(loc='upper right')

# 第2种调用方式
plt.subplot(2,2,2)
y1=stats.norm.pdf(x,loc=0,scale=2)
y2=stats.norm.pdf(x,loc=1,scale=3)
plt.plot(x, y1, color='r', label='u=0,sigma=2')
plt.plot(x, y2, color='b', label='u=0,sigma=2')
plt.legend(loc='upper right')

# stats.norm.pdf 和 stats.norm.rvs 的区别
plt.subplot(2,2,3)
y1=stats.norm.rvs(loc=0,scale=2,size=20)
y2=stats.norm.rvs(loc=1,scale=3,size=20)
plt.plot(x,y1,color='black',linestyle=':',label='u=0,sigma=2')
plt.plot(x,y2,color='purple',label='u=1,sigma=3')
plt.legend(loc='upper right')

plt.subplot(2,2,4)
y1=sorted(stats.norm.rvs(loc=0,scale=2,size=20))
y2=sorted(stats.norm.rvs(loc=1,scale=3,size=20))
plt.plot(x,y1,color='black',linestyle=':',label='u=0,sigma=2')
plt.plot(x,y2,color='purple',label='u=1,sigma=3')
plt.legend(loc='upper right')

plt.show()

