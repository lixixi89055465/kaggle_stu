# -*- coding: utf-8 -*-
# @Time : 2024/3/22 13:38
# @Author : nanji
# @Site : 
# @File : testSlice.py
# @Software: PyCharm 
# @Comment :https://blog.csdn.net/Rocky006/article/details/134207441



s = "Hello, World!"
print(s[slice(0, 5)])  # 输出 "Hello"
print(s[slice(None, 5)])  # 输出 "Hello"

print('0'*100)
nums = [1, 2, 3, 4, 5]
print(nums[slice(1, 4)])  # 输出 [2, 3, 4]
print(nums[slice(1, None)])  # 输出 [2, 3, 4, 5]
print('1'*100)
s = "Hello, World!"
print(s[slice(0, 12, 2)])  # 输出 "HloWrd"
print(s[slice(0, 12, None)])  # 输出 "Hello, World!"
print('2'*100)
nums = [1, 2, 3, 4, 5]
print(nums[slice(None, 3)])  # 输出 [1, 2, 3]
print(nums[slice(2, None)])  # 输出 [3, 4, 5]
print(nums[slice(None)])  # 输出 [1, 2, 3, 4, 5]
print('3'*100)
nums = [1, 2, 3, 4, 5]
even_nums = nums[slice(1, 5, 2)]
print(even_nums)  # 输出 [2, 4]

print('4'*100)
s = "Hello, World!"
upper_s = s[slice(None)].upper()
print(upper_s)  # 输出 "HELLO, WORLD!"

print('5'*100)
a = [1, 2, 3]
b = [4, 5, 6]
c = a[slice(None)] + b[slice(None)]
print(c)  # 输出 [1, 2, 3, 4, 5, 6]