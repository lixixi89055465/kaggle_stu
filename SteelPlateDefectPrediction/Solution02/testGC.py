# -*- coding: utf-8 -*-
# @Time : 2024/4/13 14:17
# @Author : nanji
# @Site : 
# @File : testGC.py
# @Software: PyCharm 
# @Comment :

import sys
a = "hello world"
b=sys.getrefcount(a)
print(b)