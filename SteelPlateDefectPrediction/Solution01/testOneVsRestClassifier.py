# -*- coding: utf-8 -*-
# @Time : 2024/3/25 21:35
# @Author : nanji
# @Site : https://blog.csdn.net/xdg15294969271/article/details/119850354
# @File : testOneVsRestClassifier.py
# @Software: PyCharm 
# @Comment : 

import numpy as np
from sklearn import svm,datasets
from sklearn.metrics import roc_curve,auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
import matplotlib.pyplot as plt


#加载鸢尾数据集
iris = datasets.load_iris()
X = iris.data
y = iris.target

#将数据二进制化处理，此处和onehotencoder大概一致
y = label_binarize(y,classes=[0,1,2])
n_classes = y.shape[1]

#加入噪点
n_sample,n_featrues = X.shape
X = np.c_[X,np.random.RandomState(0).randn(n_sample,80*n_featrues)]

#分割数据
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=.5,random_state=0)

#设置分类器，这里实用的是SVC支持向量机
clf =SVC(C=0.2,gamma=0.2,kernel='linear', probability=True, random_state=0)
classifier = OneVsRestClassifier(clf,n_jobs=2)
classifier.fit(X_train, y_train)
#计算分数，roc_curve要用到
y_score = classifier.decision_function(X_test)

fpr = dict()
tpr = dict()
roc_auc = dict()
#计算aoc值
for i in range(n_classes):
    fpr[i],tpr[i],_=roc_curve(y_test[:,i],y_score[:,i])
    roc_auc[i] = auc(fpr[i],tpr[i])


#画图
plt.figure()
lw = 2
color = ['r', 'g', 'b']
for i in range(3):
    plt.plot(fpr[i], tpr[i], color=color[i],lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[i])
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()

