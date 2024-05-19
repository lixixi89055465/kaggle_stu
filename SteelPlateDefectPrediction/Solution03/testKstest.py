# -*- coding: utf-8 -*-
# @Time : 2024/5/18 11:52
# @Author : nanji
# @Site : https://blog.csdn.net/qq_27782503/article/details/109291600
# @File : testKstest.py
# @Software: PyCharm 
# @Comment :
from scipy.stats import kstest
import numpy as np

x = np.random.normal(0, 1, 1000)
test_stat = kstest(x, 'norm')
# print(test_stat)

from scipy.stats import kstest
import numpy as np

x = np.random.binomial(0, 1, 1000)
test_stat = kstest(x, 'norm')
print('1' * 100)
print(test_stat)

print('2' * 100)
from scipy.stats import ks_2samp

beta = np.random.beta(7, 5, 10000)
norm = np.random.normal(0, 1, 1000)
print('3' * 100)
r2 = ks_2samp(beta, norm)
print(r2)
print('4' * 100)
from scipy.stats import ks_2samp

beta = np.random.beta(7, 5, 1000)
norm = np.random.beta(7, 5, 1000)
r3 = ks_2samp(beta, norm)
print('r3:')
print(r3)
from scipy import stats
x=[1,3,5,7,9]
y=[2,4,6,8,10]
r4=stats.wilcoxon(x,y)
print('5'*100)
print('r4:')
print(r4)

from scipy import stats
x = [1.26, 0.34, 0.70, 1.75, 50.57, 1.55, 0.08, 0.42, 0.50, 3.20, 0.15, 0.49, 0.95, 0.24, 1.37, 0.17, 6.98, 0.10, 0.94, 0.38]
y = [2.37, 2.16, 14.82, 1.73, 41.04, 0.23, 1.32, 2.91, 39.41, 0.11, 27.44, 4.51, 0.51, 4.50, 0.18, 14.68, 4.66, 1.30, 2.06, 1.19]
r5=stats.wilcoxon(x,y)
print('r5:')
print(r5)
print('6'*100)
print(stats.ttest_ind(x, y))

from scipy import stats
x = [1, 3, 5, 7, 9]
y = [2, 4, 6, 8, 10]
print('7'*100)
print(stats.kruskal(x, y))

from scipy import stats
x = [1, 3, 5, 7, 9]
y = [2, 4, 6, 8, 10]
print('8'*100)
print(stats.mannwhitneyu(x, y))



