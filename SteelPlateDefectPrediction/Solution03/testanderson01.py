# -*- coding: utf-8 -*-
# @Time : 2024/5/18 16:58
# @Author : nanji
# @Site :https://www.jianshu.com/p/7356cc80a96f#:~:text=%E5%AE%98%E6%96%B9%E6%96%87%E6%A1%A3-,scipy,-.stats.normaltest%0Ascipy
# @File : testanderson01.py
# @Software: PyCharm 
# @Comment :
import numpy as np
from scipy.stats import anderson

x = np.linspace(-15, 15, 9)
r1 = anderson(x)
print(r1)

