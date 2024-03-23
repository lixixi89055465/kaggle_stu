# -*- coding: utf-8 -*-
# @Time : 2024/3/23 下午4:57
# @Author : nanji
# @Site :
# @File : solve01.py
# @Software: PyCharm 
# @Comment : https://www.kaggle.com/code/getanmolgupta01/defect-pred-eda-xgboost-lgbm-catboost

'''

# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load
'''


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory


import os
for dirname, _, filenames in os.walk('../data/'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# Libray for Data Manipulation.
import pandas as pd
import numpy as np

#Library for Data Visualization.
import seaborn as sns
import matplotlib.pyplot as plt