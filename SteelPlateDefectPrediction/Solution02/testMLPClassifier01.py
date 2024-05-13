# -*- coding: utf-8 -*-
# @Time : 2024/4/16 13:21
# @Author : nanji
# @Site :https://pythonjishu.com/sklearn-neural_network-mlpclassifier/
# @File : testMLPClassifier01.py
# @Software: PyCharm 
# @Comment :
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix

# # 加载数据集
# iris = load_iris()
# X = iris.data
# y = iris.target
#
# # 划分训练集和测试集
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
# # 训练模型
# mlp = MLPClassifier(hidden_layer_sizes=(10, 10), max_iter=300)
# mlp.fit(X_train, y_train)
#
# # 预测结果
# predictions = mlp.predict(X_test)
# # 输出准确率和混淆矩阵
# print('0' * 100)
# print(classification_report(y_test, predictions))
# print('1' * 100)
# print(confusion_matrix(y_test, predictions))


from sklearn.datasets import fetch_openml
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 载入数据集
data, target = fetch_openml('mnist_784', version=1, return_X_y=True)
# 对数据进行归一化
data = data / 255.0
# 切分数据集合
(x_train, x_test, y_train, y_test) = train_test_split(data, target, test_size=0.2)
# 训练模型
mlp=MLPClassifier(hidden_layer_sizes=(256,128),max_iter=300)
mlp.fit(x_train,y_train)
# 测试预测准确率
predictions = mlp.predict(x_test)
accuracy = accuracy_score(y_test, predictions)
print("Accuracy:", accuracy)




