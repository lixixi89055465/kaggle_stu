# -*- coding: utf-8 -*-
# @Time : 2024/4/16 21:25
# @Author : nanji
# @Site : https://blog.csdn.net/qq_36134437/article/details/103057909
# @File : testCountVectorizer.py
# @Software: PyCharm 
# @Comment : 
from sklearn.feature_extraction.text import CountVectorizer
texts=["dog cat fish","dog cat cat","fish bird", 'bird'] # “dog cat fish” 为输入列表元素,即代表一个文章的字符串
cv = CountVectorizer()#创建词袋数据结构
cv_fit=cv.fit_transform(texts)
#上述代码等价于下面两行
#cv.fit(texts)
#cv_fit=cv.transform(texts)
print(cv.get_feature_names())    #['bird', 'cat', 'dog', 'fish'] 列表形式呈现文章生成的词典
print(cv.vocabulary_	)              # {‘dog’:2,'cat':1,'fish':3,'bird':0} 字典形式呈现，key：词，value:词频
print(cv_fit)
print(cv_fit.toarray()) #.toarray() 是将结果转化为稀疏矩阵矩阵的表示方式；
print(cv_fit.toarray().sum(axis=0))  #每个词在所有文档中的词频



