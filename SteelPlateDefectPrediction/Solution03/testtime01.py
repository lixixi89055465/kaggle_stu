# -*- coding: utf-8 -*-
# @Time : 2024/5/22 11:27
# @Author : nanji
# @Site : 
# @File : testtime01.py
# @Software: PyCharm 
# @Comment : 
import warnings

warnings.filterwarnings('ignore')
import time

def get_time(f):
    def inner(*arg,**kwarg):
        s_time = time.time()
        res = f(*arg,**kwarg)
        e_time = time.time()
        print('耗时：{}秒'.format(e_time - s_time))
        return res
    return inner

@get_time
def atest():
    time.sleep(2)  # 模拟运行2s

# for i in range(3):
#     atest()

from datetime import datetime

print('0'*100)
print(datetime.now())
