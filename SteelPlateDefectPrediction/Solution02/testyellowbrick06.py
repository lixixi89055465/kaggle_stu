# -*- coding: utf-8 -*-
# @Time : 2024/4/13 22:19
# @Author : nanji
# @Site : https://blog.csdn.net/LuohenYJ/article/details/107575950
# @File : testyellowbrick06.py
# @Software: PyCharm 
# @Comment : 
from sklearn.cluster import KMeans
from yellowbrick.cluster.elbow import kelbow_visualizer
from yellowbrick.datasets.loaders import load_nfl

X, y = load_nfl()

# Use the quick method and immediately show the figure
kelbow_visualizer(KMeans(random_state=4), X, k=(2,10));
