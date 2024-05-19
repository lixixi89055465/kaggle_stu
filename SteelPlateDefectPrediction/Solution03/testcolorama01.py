# -*- coding: utf-8 -*-
# @Time : 2024/5/18 20:57
# @Author : nanji
# @Site :https://zhuanlan.zhihu.com/p/682742754
# @File : testcolorama01.py
# @Software: PyCharm 
# @Comment :
from colorama import Fore, Back, Style

print(Fore.RED + '这是一段红色文本')
print(Back.GREEN + '并有一个绿色背景')
print(Style.DIM + '文字变得暗淡')
print(Style.RESET_ALL)
print('现在又回到了正常文本')
# from colorama import Fore, Back,init
# import colorama
#
# custom_color =init(255, 0, 255)  # 紫色
#
# print(custom_color + "这是自定义紫色文本")