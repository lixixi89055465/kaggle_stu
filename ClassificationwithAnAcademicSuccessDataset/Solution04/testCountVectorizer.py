# -*- coding: utf-8 -*-
# @Time : 2024/10/19 10:22
# @Author : nanji
# @Site : https://www.zhihu.com/tardis/zm/art/268886634?source_id=1005
# @File : testCountVectorizer.py
# @Software: PyCharm 
# @Comment :
from sklearn.feature_extraction.text import CountVectorizer

# # 语料
# corpus = [
#     'This is the first document.',
#     'This is the this second second document.',
#     'And the third one.',
#     'Is this the first document?'
# ]
# vectorizer = CountVectorizer()
# print(vectorizer)
# # 计算某个词出现的次数
# X = vectorizer.fit_transform(corpus)
# print(type(X), X)
# # 获取词袋中所有文本关键词
# word = vectorizer.get_feature_names()
# print('1'*100)
#
# print(word)
#
# #查看词频结果
# print('2'*100)
#
# print(X.toarray())

from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer, CountVectorizer
import numpy as np

# # 语料
cc = [
    'aa bb.',
    'aa cc.'
]
# # method1
# vectorizer = TfidfVectorizer()
# X = vectorizer.fit_transform(cc)
# print('feature ', vectorizer.get_feature_names())
# print(X.toarray())
# method 2
vectorizer=CountVectorizer()#token_pattern='(?u)\\b\\w+\\b'
transformer = TfidfTransformer()
cntTf = vectorizer.fit_transform(cc)
# print('feature', vectorizer.get_feature_names())
# print(cntTf)
cnt_array = cntTf.toarray()
X = transformer.fit_transform(cntTf)
# print(X.toarray())

# method 3
vectorizer=CountVectorizer()
cntTf = vectorizer.fit_transform(cc)
tf = cnt_array/np.sum(cnt_array, axis = 1, keepdims = True)
# print('1'*100)

# print('tf',tf)
idf = np.log((1+len(cnt_array))/(1+np.sum(cnt_array,axis = 0))) + 1
# print('idf', idf)
t = tf*idf
# print('tfidf',t)
# print('norm tfidf', t/np.sqrt(np.sum(t**2, axis = 1, keepdims=True)))



print('2'*100)
from sklearn.feature_extraction.text import HashingVectorizer
corpus = [
     'This is the first document.',
     'This document is the second document.',
     'And this is the third one.',
     'Is this the first document?',
 ]
vectorizer = HashingVectorizer(n_features=2**4)
X = vectorizer.fit_transform(corpus)
print(X.toarray())
print(X.shape)
