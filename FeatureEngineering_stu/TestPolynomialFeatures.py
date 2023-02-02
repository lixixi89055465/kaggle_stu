'''

'''
from sklearn.preprocessing import PolynomialFeatures
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

X = np.arange(6).reshape(3, 2)
# print(X)
poly = PolynomialFeatures(2)  # degree =2 ,默认interaction_only=False
# print(poly.fit_transform(X))
# print('0' * 100)
# print(poly.powers_)

# 默认的阶数为2，同时设置交互关系为true
poly = PolynomialFeatures(interaction_only=True, include_bias=False)
# print('1' * 100)
# print(poly.fit_transform(X))
# print('2' * 100)
# print(poly.powers_)
# print('3'*100)
# print(poly.n_input_features_)

# x从 -3-3 均匀取值
# x从-3 - 3均匀取值
x = np.random.uniform(-3, 3, size=100)
X = x.reshape(-1, 1)
# y是二次方程
y = 0.5 * x ** 2 + x + 2 + np.random.normal(0, 1, size=100)
plt.scatter(x, y)
plt.show()
# 实例化线形模型
from sklearn.linear_model import LinearRegression

# lr = LinearRegression()
# lr.fit(X, y)
# y_predict = lr.predict(X)
# plt.scatter(x, y)
# plt.plot(x, y_predict)
# plt.show()
#
# poly = PolynomialFeatures(degree=2)
# degree=2 生成2次特征，可以调整
# poly.fit(X)
# X2 = poly.transform(X)
# print('x的大小:', X2.shape)
# print(X2[:5, ])
# 继续使用线形模型
# lr.fit(X2, y)
# y_predict2 = lr.predict(X2)
# plt.scatter(x, y)
# plt.plot(np.sort(x), y_predict2[np.argsort(x)])
# plt.show()

X = np.sort(3 * np.random.rand(40, 1), axis=0)
y = np.sin(X).ravel()
# 噪声
y[::5] += 2.5 * (0.5 - np.random.rand(8))
plt.plot(X, y, 'b^')
plt.show()
lr = LinearRegression()
pf = PolynomialFeatures(degree=2)
lr.fit(pf.fit_transform(X), y)
print(lr.coef_)
print(lr.intercept_)
xx = np.linspace(0, 5, 100)
xx2 = pf.transform(xx[:, np.newaxis])

yy2 = lr.predict(xx2)
plt.plot(xx, yy2, c='r')
plt.show()
