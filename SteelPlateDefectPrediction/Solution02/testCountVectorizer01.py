# -*- coding: utf-8 -*-
# @Time : 2024/4/16 21:06
# @Author : nanji
# @Site :https://zhuanlan.zhihu.com/p/166636681
# @File : testCountVectorizer01.py
# @Software: PyCharm 
# @Comment : 
from sklearn.feature_extraction.text import CountVectorizer

corpus = [
	'This is the first document.',
	'This document is the second document.',
	'And this is the third one.',
	'Is this the first document?',
]
vectorizer = CountVectorizer()
# 将文本数据转换为计数的稀疏矩阵
X = vectorizer.fit_transform(corpus)
print(X.shape)
# 输出为 ['and', 'document', 'first', 'is', 'one', 'second', 'the', 'third', 'this']

# 由于 X 存储为稀疏矩阵，需要转换为 array 才能查看
print(X.toarray())
# 输出为
from sklearn.feature_extraction.text import TfidfTransformer

transform = TfidfTransformer()
Y = transform.fit_transform(X)
print('1' * 100)
print(Y.toarray())
# 把每个设备的 app 列表转换为字符串，以空格分隔
# apps = deviceid_packages['apps'].apply(lambda x: ' '.join(x)).tolist()
# vectorizer = CountVectorizer()
# transformer = TfidfTransformer()
# # 原来的 app 列表 转换为计数的稀疏矩阵。
# cntTf = vectorizer.fit_transform(apps)
# # 得到 tf-idf 矩阵
# tfidf = transformer.fit_transform(cntTf)
# # 得到所有的 APP 列表，相当于词典
# word = vectorizer.get_feature_names()
