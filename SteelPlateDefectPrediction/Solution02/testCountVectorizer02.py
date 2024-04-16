# -*- coding: utf-8 -*-
# @Time : 2024/4/16 21:41
# @Author : nanji
# @Site : https://blog.csdn.net/qq_36134437/article/details/103057909
# @File : testCountVectorizer02.py
# @Software: PyCharm 
# @Comment :
from sklearn.feature_extraction.text import CountVectorizer

# 语料
corpus = [
	'This is the first document.',
	'This is the this second second document.',
	'And the third one.',
	'Is this the first document?'
]

#将文本中的词转换成词频矩阵
vectorizer = CountVectorizer()
print(vectorizer)

#计算某个词出现的次数
X = vectorizer.fit_transform(corpus)
print(type(X),X)

#获取词袋中所有文本关键词
word = vectorizer.get_feature_names()
print(word)
#查看词频结果
print(X.toarray())