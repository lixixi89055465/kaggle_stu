# -*- coding: utf-8 -*-
# @Time : 2024/4/16 21:50
# @Author : nanji
# @Site : https://blog.csdn.net/qq_36134437/article/details/103057909
# @File : testCountVectorizer03.py
# @Software: PyCharm 
# @Comment : 
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer

transformer = TfidfTransformer()
print(transformer)
corpus = [
	'This is the first document.',
	'This is the this second second document.',
	'And the third one.',
	'Is this the first document?'
]

# 将文本中的词转换成词频矩阵
vectorizer = CountVectorizer()
print(vectorizer)
# 计算某个词出现的次数
X = vectorizer.fit_transform(corpus)
tfidf = transformer.fit_transform(X)
print(tfidf.toarray())
