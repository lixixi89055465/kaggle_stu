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
x=np.random.binomial(0,1,1000)
test_stat=kstest(x,'norm')
print('1'*100)
print(test_stat)

print('2'*100)
from scipy.stats import ks_2samp
beta=np.random.beta

