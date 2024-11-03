# -*- coding: utf-8 -*-
# @Time : 2024/11/3 13:32
# @Author : nanji
# @site : https://vimsky.com/examples/usage/python-sklearn.preprocessing.powertransformer-sk.html
# @File : testPowerTransformer.py
# @Software: PyCharm 
# @Comment :

import numpy as np
from sklearn.preprocessing import PowerTransformer

pt = PowerTransformer()
data = [
    [1, 2], [3, 2], [4, 5]
]
print(pt.fit(data))
print('0' * 100)
print(pt.lambdas_)
print('1' * 100)
pt.transform(data)
