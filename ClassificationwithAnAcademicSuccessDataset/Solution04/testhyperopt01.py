# -*- coding: utf-8 -*-
# @Time : 2024/10/29 22:00
# @Author : nanji
# @Site : https://zhuanlan.zhihu.com/p/276601891
# @File : testhyperopt01.py
# @Software: PyCharm 
# @Comment : https://zhuanlan.zhihu.com/p/276601891
from hyperopt import fmin, tpe, hp,Trials

trials = Trials()

best = fmin(fn=lambda x: x ** 2,
            space= hp.uniform('x', -10, 10),
            algo=tpe.suggest,
            max_evals=500,
            trials = trials)

print(best)