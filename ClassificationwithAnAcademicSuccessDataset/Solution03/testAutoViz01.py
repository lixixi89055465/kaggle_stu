# -*- coding: utf-8 -*-
# @Time : 2024/10/13 21:33
# @Author : nanji
# @Site : https://zhuanlan.zhihu.com/p/416302074
# @File : testAutoViz01.py
# @Software: PyCharm 
# @Comment :

from autoviz.AutoViz_Class import AutoViz_Class

# AutoViz实例化
AV = AutoViz_Class()
# 一行代码实现数据探索
dft = AV.AutoViz(
    filename='../input/iris.csv',  # 读入数据集，注意和dfte的区别
    sep=',',  # 设置数据集分隔符+，默认未逗号，
    depVar='species',  # 设置因变量
    dfte=None,  # 传入一个pandas+.DataFrame,如果filename已设置，此处为None,反之亦然
    header=0,
    verbose=0,  # 可选0,1,或2,设置图形的保存形式
    lowess=False,  # 是否启用Lowess回归，适合小数据量数据集，100,000行以上数据不建议用
    chart_format='svg',  # 设置图形保存格式
    max_rows_analyzed=150000,  # 设置数据集待分析的行数
    max_cols_analyzed=30,  # 设置数据集待分析的列数
)
# 结果输出一部分为Dataset的简单介绍
# 结果输出另一部分为大量可视化图表


