# -*- coding: utf-8 -*-
# @Time : 2024/4/13 22:21
# @Author : nanji
# @Site : https://blog.csdn.net/LuohenYJ/article/details/107575950
# @File : testyellowbrick.py
# @Software: PyCharm 
# @Comment : 
# from sklearn.cluster import KMeans
# from yellowbrick.cluster import SilhouetteVisualizer
# from yellowbrick.datasets import load_nfl
#
# X,y=load_nfl()
# features = ['Rec', 'Yds', 'TD', 'Fmb', 'Ctch_Rate']
# print('0'*100)
# print(X.columns)
# X = X.query('Tgt >= 20')[features]
# model = KMeans(5, random_state=42)
# visualizer = SilhouetteVisualizer(model, colors='yellowbrick')
# visualizer.fit(X)        # Fit the data to the visualizer
# visualizer.show();        # Finalize and render the figure

from sklearn.cluster import KMeans

from yellowbrick.cluster import silhouette_visualizer
from yellowbrick.datasets import load_credit

# Load a clustering dataset
X = load_credit()

# Specify rows to cluster: under 40 y/o and have either graduate or university education
X = X[(X['age'] <= 40) & (X['edu'].isin([1,2]))]

# Use the quick method and immediately show the figure
silhouette_visualizer(KMeans(5, random_state=42), X, colors='yellowbrick');
