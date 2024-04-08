# -*- coding: utf-8 -*-
# @Time : 2024/4/8 11:36
# @Author : nanji
# @Site : https://www.jianshu.com/p/49ab87122562
# @File : testcatboost.py
# @Software: PyCharm 
# @Comment :
from catboost import CatBoostClassifier
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
data  = pd.read_csv("ctr_train.txt", delimiter="\t")
del data["user_tags"]
data = data.fillna(0)
X_train, X_validation, y_train, y_validation = train_test_split(data.iloc[:,:-1],data.iloc[:,-1],test_size=0.3 , random_state=1234)

