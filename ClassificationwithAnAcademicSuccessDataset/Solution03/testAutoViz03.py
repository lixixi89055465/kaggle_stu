# -*- coding: utf-8 -*-
# @Time : 2024/10/13 22:21
# @Author : nanji
# @Site : 
# @File : testAutoViz03.py
# @Software: PyCharm 
# @Comment :

from sklearn.datasets import load_iris

# 加载 Iris 数据集
iris = load_iris()
X, y = iris.data, iris.target

import pandas as pd

# 将数据集转换为 DataFrame
df = pd.DataFrame(X, columns=iris.feature_names)
df['target'] = y

from autoviz.AutoViz_Class import AutoViz_Class

# 创建 Autoviz 实例
AV = AutoViz_Class()

# 创建可视化图表
report = AV.AutoViz(df)
# 创建可视化图表
report = AV.AutoViz(df, depVar='target')