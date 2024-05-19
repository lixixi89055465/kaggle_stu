# -*- coding: utf-8 -*-
# @Time : 2024/5/18 16:01
# @Author : nanji
# @Site :https://zhuanlan.zhihu.com/p/410700123
# @File : testnormaltest01.py
# @Software: PyCharm 
# @Comment :
from scipy import stats

import numpy as np
from scipy.stats import kstest

x = stats.norm.rvs(loc=0.2, size=100)
r1 = kstest(x, 'norm')
print('1'*100)
print(r1)

from scipy.stats import normaltest
r2=normaltest(x)
print('2'*100)
print(r2)
