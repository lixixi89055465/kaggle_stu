# -*- coding: utf-8 -*-
# @Time : 2024/10/19 9:55
# @Author : nanji
# @Site : https://www.cnblogs.com/Luv-GEM/p/10888026.html
# @File : test_Tf-Idf.py
# @Software: PyCharm 
# @Comment :
import os, re
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics

import nltk
from nltk.corpus import stopwords

datafile = os.path.join('..', 'data', 'labeledTrainData.tsv')
# escapechar='\\'用来去掉转义字符'\'
df = pd.read_csv(datafile, sep='\t', escapechar='\\')
print('Number of reviews: {}'.format(len(df)))
print(df.head())