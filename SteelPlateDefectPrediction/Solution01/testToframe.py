# -*- coding: utf-8 -*-
# @Time : 2024/3/24 9:54
# @Author : nanji
# @Site : https://vimsky.com/examples/usage/python-pandas-series-to_frame.html
# @File : testToframe.py
# @Software: PyCharm 
# @Comment :

import pandas as pd

# Creating the Series
sr = pd.Series(['New York', 'Chicago', 'Toronto', 'Lisbon', 'Rio', 'Moscow'])

# Create the Datetime Index
didx = pd.DatetimeIndex(start='2014-08-01 10:00', freq='W',
						periods=6, tz='Europe/Berlin')

# set the index
sr.index = didx

# Print the series
print(sr)