# -*- coding: utf-8 -*-
# @Time : 2024/10/13 22:39
# @Author : nanji
# @Site : 
# @File : testAutoViz04.py
# @Software: PyCharm 
# @Comment :https://github.com/snewton17/AutoViz

import pandas as pd
from autoviz import AutoViz_Class

AV = AutoViz_Class()

data = {'col1': [1, 2, 3, 4, 5], 'col2': [5, 4, 3, 2, 1]}
df = pd.DataFrame(data)

dft = AV.AutoViz(
    "",
    sep=",",
    depVar="",
    dfte=df,
    header=0,
    verbose=1,
    lowess=False,
    chart_format="server",
    max_rows_analyzed=150000,
    max_cols_analyzed=30,
    save_plot_dir=None
)