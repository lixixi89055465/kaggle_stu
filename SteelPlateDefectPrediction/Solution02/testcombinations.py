# -*- coding: utf-8 -*-
# @Time    : 2024/4/13 上午12:22
# @Author  : nanji
# @Site    : https://blog.csdn.net/u011699626/article/details/134513874
# @File    : testcombinations.py
# @Software: PyCharm 
# @Comment :

from itertools import combinations

results = combinations("ABCD", 2)
for result in results:
    print(result)
"""
result:
('A', 'B')
('A', 'C')
('A', 'D')
('B', 'C')
('B', 'D')
('C', 'D')
"""
print('1'*100)
from itertools import combinations

results = combinations(range(4), 3)
for result in results:
    print(result)
"""
result:
(0, 1, 2)
(0, 1, 3)
(0, 2, 3)
(1, 2, 3)
"""
