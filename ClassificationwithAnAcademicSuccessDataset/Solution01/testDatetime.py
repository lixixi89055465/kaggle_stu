# -*- coding: utf-8 -*-
# @Time : 2024/6/14 11:29
# @Author : nanji
# @Site : 
# @File : testDatetime.py
# @Software: PyCharm 
# @Comment : 
import warnings

warnings.filterwarnings('ignore')
from datetime import datetime

print(datetime.now().date())
print(datetime.now().time())
print(datetime.now().strftime("%Y%m%d-%H-%M-%S"))
