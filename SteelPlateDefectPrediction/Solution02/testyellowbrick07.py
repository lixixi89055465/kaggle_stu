# -*- coding: utf-8 -*-
# @Time : 2024/4/14 9:42
# @Author : nanji
# @Site : https://blog.csdn.net/LuohenYJ/article/details/107575950
# @File : testyellowbrick07.py
# @Software: PyCharm 
# @Comment :

from sklearn.cluster import KMeans
from yellowbrick.cluster import SilhouetteVisualizer
from yellowbrick.datasets import load_nfl
X,y=load_nfl()
print('0'*100)
print(X.columns)
features = ['Rec', 'Yds', 'TD', 'Fmb', 'Ctch_Rate']
X=X.query('Tgt >= 20')[features]
model=KMeans(5,random_state=42)
visualizer=SilhouetteVisualizer(model,color='yellowbrick')

visualizer.fit(X)
visualizer.show()

