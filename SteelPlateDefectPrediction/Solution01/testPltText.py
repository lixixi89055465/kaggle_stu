# -*- coding: utf-8 -*-
# @Time : 2024/3/30 16:36
# @Author : nanji
# @Site : https://blog.csdn.net/TeFuirnever/article/details/88947248
# @File : testPltText.py
# @Software: PyCharm 
# @Comment :

import matplotlib.pyplot as plt

fig = plt.figure()
plt.axis([0, 10, 0, 10])
t = "This is a really long string that I'd rather have wrapped so that it"\
    " doesn't go outside of the figure, but if it's long enough it will go"\
    " off the top or bottom!"
plt.text(4, 1, t, ha='left', rotation=15, wrap=True)
plt.text(6, 5, t, ha='left', rotation=15, wrap=True)
plt.text(6, 5, t, ha='left', rotation=15, wrap=False)
plt.show()
