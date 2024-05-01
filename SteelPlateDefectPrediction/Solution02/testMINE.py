# -*- coding: utf-8 -*-
# @Time : 2024/5/1 20:33
# @Author : nanji
# @Site : https://zhuanlan.zhihu.com/p/141506312
# @File : testMINE.py
# @Software: PyCharm 
# @Comment : 
import numpy as np
from minepy import MINE
m = MINE()
x = np.random.uniform(-1, 1, 10000)
m.compute_score(x, x**2)
print(m.mic())