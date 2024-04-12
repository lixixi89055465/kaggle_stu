# -*- coding: utf-8 -*-
# @Time : 2024/4/11 22:07
# @Author : nanji
# @Site : https://zhuanlan.zhihu.com/p/206597487
# @File : testsweetviz.py
# @Software: PyCharm 
# @Comment : 
import pandas as pd
import sweetviz as sv

data=pd.read_csv("./midwest.csv")
midwest_report = sv.analyze(data)
midwest_report.show_html()

A=data.query("category in ('AAR', 'AAU')")
B=data.query("category not in ('AAR', 'AAU')")
sv.compare([A, 'A'], [B, 'B']).show_html()