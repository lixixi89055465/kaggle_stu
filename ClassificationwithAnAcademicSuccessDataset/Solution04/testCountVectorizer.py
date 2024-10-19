# -*- coding: utf-8 -*-
# @Time : 2024/10/19 10:22
# @Author : nanji
# @Site : https://www.zhihu.com/tardis/zm/art/268886634?source_id=1005
# @File : testCountVectorizer.py
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
vectorizer = CountVectorizer()
print(vectorizer)
# 计算某个词出现的次数
X = vectorizer.fit_transform(corpus)
print(type(X), X)
# 获取词袋中所有文本关键词
word = vectorizer.get_feature_names()
print('1'*100)

print(word)

#查看词频结果
print('2'*100)

print(X.toarray())
