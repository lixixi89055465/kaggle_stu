# -*- coding: utf-8 -*-
# @Time : 2024/11/2 23:18
# @Author : nanji
# @Site : 
# @File : testcrosstab.py
# @Software: PyCharm 
# @Comment :
import pandas as pd

# 创建DataFrame
data = {
    "用户ID": [1, 1, 2, 3, 3],
    "购买时间": ["2023-07-01", "2023-07-02", "2023-07-01", "2023-07-02", "2023-07-02"],
    "购买商品类别": ["食品", "饮料", "饮料", "食品", "饮料"]
}

df = pd.DataFrame(data)

# 创建交叉表
crosstab_table = pd.crosstab(df["用户ID"], df["购买商品类别"])

# 打印交叉表
print(crosstab_table)
