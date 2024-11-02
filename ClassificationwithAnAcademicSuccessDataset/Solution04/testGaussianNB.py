# -*- coding: utf-8 -*-
# @Time : 2024/11/2 23:36
# @Author : nanji
# @Site : https://blog.csdn.net/aouiylfjh/article/details/131535742
# @File : testGaussianNB.py
# @Software: PyCharm 
# @Comment :
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import numpy as np
from scipy.stats import norm
from sklearn.metrics import accuracy_score


class MYGaussianNB:
    def fit(self, X, y):
        #训练数据为一个二维数组，其中每一行是一个样本的特征向量。标签是一个一维数组
        self.classes = np.unique(y)                      # 类别
        self.class_priors = np.zeros(len(self.classes))  # 每个类别的先验概率
        self.means = {}                                  # 每个类别都有个array储存每个特征的均值
        self.variances = {}                              # 每个类别都有个array储存每个特征的方差

        for i, class_name in enumerate(self.classes):
            class_indices = y == class_name              # 为True的索引表示为是该类的样本所在的索引
            class_X = X[class_indices]                   # 根据索引获取该类所有样本
            self.class_priors[i] = len(class_X) / len(X) # 获取该类的先验概率
            self.means[class_name] = np.mean(class_X, axis=0) # 求出该类样本所有特征下的均值
            self.variances[class_name] = np.var(class_X, axis=0)# 求出该类样本所有特征下的方差

    def predict(self, X):
        y_pred = []  # 创建一个空列表，用于存储预测结果
        for x in X:  # 对输入的每个样本进行遍历
            class_scores = []  # 创建一个空列表，用于存储每个类别的得分
            for i, class_name in enumerate(self.classes): # 遍历每个类别和其索引
                prior = self.class_priors[i]              # 获取当前类别的先验概率
                mean = self.means[class_name]             # 获取当前类别的均值
                variance = self.variances[class_name]     # 获取当前类别的方差
                likelihood = norm.pdf(x, loc=mean, scale=np.sqrt(variance))
                                                          # 计算当前样本在当前类别下的似然概率
                score = prior * np.prod(likelihood)       # 计算当前样本在当前类别下的得分（先验概率乘以似然概率）
                class_scores.append(score)                # 将当前类别的得分添加到列表中
            y_pred.append(self.classes[np.argmax(class_scores)])  # 将具有最高得分的类别作为预测结果添加到列表中
        return y_pred  # 返回预测结果列表


# 加载燕尾花数据集
iris = load_iris()

# 获取特征数据和标签数据
X = iris.data
y = iris.target

# 将数据集切分为训练集和测试集
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=3220822)

# 打印切分后的数据集大小
print("训练集大小:", x_train.shape[0])
print("测试集大小:", x_test.shape[0])
from sklearn.naive_bayes import GaussianNB
# 创建高斯朴素贝叶斯模型对象
model = GaussianNB()
# 模型训练
model.fit(x_train, y_train)
# 预测
y_pred = model.predict(x_test)
from sklearn.metrics import accuracy_score
# 评估模型
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
# 创建 GaussianNB 分类器并训练
classifier = MYGaussianNB()
classifier.fit(x_train, y_train)

# 进行预测
y_pred = classifier.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
accuracy
