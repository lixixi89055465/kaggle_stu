# -*- coding: utf-8 -*-
# @Time : 2024/11/3 22:29
# @Author : nanji
# @Site : https://vimsky.com/examples/usage/python-sklearn.compose.ColumnTransformer-sk.html
# @File : testColumnTransformer03.py
# @Software: PyCharm 
# @Comment :
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import Normalizer

ct = ColumnTransformer([('norm1':Normalizer(norm='l1'), [0, 1]), ('norm2':Normalizer(norm='l1'), slice(2, 4))] )
