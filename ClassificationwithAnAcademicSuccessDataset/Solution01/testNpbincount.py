# -*- coding: utf-8 -*-
# @Time : 2024/6/6 13:39
# @Author : nanji
# @Site : 
# @File : testNpbincount.py
# @Software: PyCharm 
# @Comment : 
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')
nums = [1, 2, 3, 4, 4]
counts = np.bincount(nums)
print(counts)
