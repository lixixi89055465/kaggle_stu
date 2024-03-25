# -*- coding: utf-8 -*-
# @Time : 2024/3/25 22:12
# @Author : nanji
# @Site :  https://blog.csdn.net/weixin_42196948/article/details/123529305
# @File : testKerasClassifier.py
# @Software: PyCharm 
# @Comment :
from sklearn import datasets
import numpy as np

from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import KFold

dataset=datasets.load_iris()
x=dataset.data
Y=dataset.target
seed=7
np.random.seed(seed)
def create_model(optimizer='adam',init='glorot_uniform'):
	#构建模型
	model=Sequential()
	model.add(Dense(units=4,activation='relu',input_dim=4,kernel_initializer=init))
	model.add(Dense(units=6,activation='relu',kernel_initializer=init))
	model.add(Dense(units=3,activation='softmax',kernel_initializer=init))
	model.compile(loss='categorical_crossentropy',\
				  optimizer=optimizer,\
				  metrics=['accuracy'])
	return model

model=KerasClassifier(build_fn=create_model,epochs=200,batch_size=5,verbose=0 )
kfold=KFold(n_splits=10,shuffle=True,random_state=seed)
results=cross_val_score(model,x,Y,cv=kfold)
print('Accuracy: %.2f%% (%.2f)' % (results.mean()*100, results.std()))







