# -*- coding: utf-8 -*-
# @Time : 2024/10/13 22:01
# @Author : nanji
# @Site : 
# @File : testAutoViz02.py
# @Software: PyCharm 
# @Comment :https://blog.csdn.net/Trb401012/article/details/136282016

# 导入绝世秘籍——自动可视化心法
from autoviz.AutoViz_Class import AutoViz_Class
import pandas as pd

# 解读玄妙数据，此处以一卷CSV经文为例
df = pd.read_csv("../input/iris.csv")

# 施展绝学，启动AutoViz之道
AV = AutoViz_Class()

# 运用武林绝学，一招搞定数据探秘
# table_AV = AV.AutoViz("mystery_data.csv")  # 注：此处的"mystery_data.csv"需替换为实际的文件路径
table_AV = AV.AutoViz(filename="../input/iris.csv",save_plot_dir=None)  # 注：此处的"mystery_data.csv"需替换为实际的文件路径

print('1'*100)

