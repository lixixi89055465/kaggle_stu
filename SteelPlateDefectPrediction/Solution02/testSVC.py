# -*- coding: utf-8 -*-
# @Time : 2024/4/15 10:46
# @Author : nanji
# @Site : https://blog.csdn.net/weixin_43746433/article/details/97808078
# @File : testSVC.py
# @Software: PyCharm 
# @Comment :

'''
SVC参数解释
（1）C: 目标函数的惩罚系数C，用来平衡分类间隔margin和错分样本的，default C = 1.0；
（2）kernel：参数选择有RBF, Linear, Poly, Sigmoid, 默认的是"RBF";
（3）degree：if you choose 'Poly' in param 2, this is effective, degree决定了多项式的最高次幂；
（4）gamma：核函数的系数('Poly', 'RBF' and 'Sigmoid'), 默认是gamma = 1 / n_features;
（5）coef0：核函数中的独立项，'RBF' and 'Poly'有效；
（6）probablity: 可能性估计是否使用(true or false)；
（7）shrinking：是否进行启发式；
（8）tol（default = 1e - 3）: svm结束标准的精度;
（9）cache_size: 制定训练所需要的内存（以MB为单位）；
（10）class_weight: 每个类所占据的权重，不同的类设置不同的惩罚参数C, 缺省的话自适应；
（11）verbose: 跟多线程有关，不大明白啥意思具体；
（12）max_iter: 最大迭代次数，default = 1， if max_iter = -1, no limited;
（13）decision_function_shape ： ‘ovo’ 一对一, ‘ovr’ 多对多  or None 无, default=None
（14）random_state ：用于概率估计的数据重排时的伪随机数生成器的种子。
 ps：7,8,9一般不考虑。

————————————————

                            版权声明：本文为博主原创文章，遵循 CC 4.0 BY-SA 版权协议，转载请附上原文出处链接和本声明。

原文链接：https://blog.csdn.net/weixin_43746433/article/details/97808078

'''
# from sklearn.svm import SVC
# import numpy as np
# X = np.array([[-1, -1], [-2, -1], [1, 1], [2, 1]])
# y = np.array([1, 1, 2, 2])
#
# clf = SVC()
# clf.fit(X, y)
# print(clf.fit(X, y))
# print('2' * 100)
# print(clf.predict([[-0.8, -1]]))
#
# import numpy as np
#
# X = np.array([[-1, -1], [-2, -1], [1, 1], [2, 1]])
# y = np.array([1, 1, 2, 2])
# from sklearn.svm import NuSVC
#
# clf = NuSVC()
# clf.fit(X, y)
# print('3'*100)
# print(clf.predict([[-0.8, -1]]))


# from sklearn.svm import LinearSVC
#
# X = [[0], [1], [2], [3]]
# Y = [0, 1, 2, 3]
#
# clf = LinearSVC(decision_function_shape='ovo')  # ovo为一对一
# clf.fit(X, Y)
# print(clf.fit(X, Y))
#
# dec = clf.decision_function([[1]])  # 返回的是样本距离超平面的距离
# print(dec)
#
# clf.decision_function_shape = "ovr"
# dec = clf.decision_function([1])  # 返回的是样本距离超平面的距离
# print(dec)
# print(clf.predict([1]))
from sklearn.datasets import fetch_lfw_people
import  matplotlib.pyplot as plt
from sklearn.svm import SVC
faces = fetch_lfw_people(min_faces_per_person=60)
print(faces.target_names)
print(faces.images.shape)
fig, ax = plt.subplots(3, 5)
for i, axi in enumerate(ax.flat):
    axi.imshow(faces.images[i], cmap='bone')
    axi.set(xticks=[], yticks=[],
    xlabel=faces.target_names[faces.target[i]])


from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
pca=PCA(n_components=150,whiten=True,random_state=42,)
svc=SVC(kernel='rbf',class_weight='balanced')
model=make_pipeline(pca,svc)
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(faces.data,\
                                               faces.target,\
                                               random_state=42 )

from sklearn.model_selection import GridSearchCV
param_grid={
    'svc__c':[1,5,10],
    'svc__gamma':[0.0001,0.0005,0.0001]
}
grid=GridSearchCV(model,param_grid)

grid.fit(X_train,y_train )
print('1'*100)
print(grid.best_params_)
model=grid.best_estimator_
yfit=model.predict(X_test)
print('2'*100)
print(yfit.shape)
