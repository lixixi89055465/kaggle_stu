# -*- coding: utf-8 -*-
# @Time : 2024/10/14 16:16
# @Author : nanji
# @Site : 
# @File : testSkew.py
# @Software: PyCharm 
# @Comment :

import pandas as pd

test = pd.DataFrame({'A': [1, 2, 3, 4], 'B': [2, 4, 5, 2]})
a = test.skew(axis=0)
test['C'] = [4, 5, 6, 7]
a = test.skew(axis=0)
print(a)
