# -*- coding: utf-8 -*-
# @Time : 2024/5/1 18:00
# @Author : nanji
# @Site :https://zhuanlan.zhihu.com/p/141506312
# @File : testSelectKBest02.py
# @Software: PyCharm 
# @Comment : 
from sklearn.feature_selection import SelectKBest, f_classif

X = [
	[1, 2, 3, 4, 5],
	[5, 4, 3, 2, 1],
	[3, 3, 3, 3, 3],
	[1, 1, 1, 1, 1]
]
y = [0, 1, 0, 1]

print('before transform:\n', X)
sel = SelectKBest(score_func=f_classif, k=3)
sel.fit(X, y)
print('score_ : \n', sel.scores_)
print('pvalues_ :', sel.scores_)
print('selected index:', sel.get_support(True))
print('after transform :\n', sel.transform(X))


