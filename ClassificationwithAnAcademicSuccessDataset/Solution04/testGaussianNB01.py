# -*- coding: utf-8 -*-
# @Time : 2024/11/3 0:22
# @Author : nanji
# @Site : https://blog.csdn.net/weixin_42878111/article/details/137239571
# @File : testGaussianNB01.py
# @Software: PyCharm 
# @Comment :
import numpy as np
from sklearn.naive_bayes import GaussianNB
import torch

# 假设我们有如下数据
X_train = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
y_train = np.array([0, 0, 1, 1])

# 使用sklearn的GaussianNB训练模型
gnb = GaussianNB()
gnb.fit(X_train, y_train)

# 定义一个函数将预测过程封装起来，以便于处理PyTorch Tensors
def predict_gnb(x):
    x = x.numpy()  # 将Tensor转化为numpy数组
    return gnb.predict(x)

# 创建一个PyTorch Tensor作为测试数据
X_test_torch = torch.tensor([[9, 10], [11, 12]])

# 使用封装好的predict函数进行预测
predictions = predict_gnb(X_test_torch)
print(predictions)
