import numpy as np
import matplotlib.pyplot as plt
# X = np.array([[0, 0.5, 1],[0, 0.5, 1]])
# print("X的维度:{},shape:{}".format(X.ndim, X.shape))
# Y = np.array([[0, 0, 0],[1, 1, 1]])
# print("Y的维度:{},shape:{}".format(Y.ndim, Y.shape))
#
# plt.plot(X, Y, 'o--')
# plt.grid(True)
# plt.show()

# x = np.array([0, 0.5, 1])
# y = np.array([0, 1])
#
# xv, yv = np.meshgrid(x, y)
# print("xv的维度:{},shape:{}".format(xv.ndim, xv.shape))
# print("yv的维度:{},shape:{}".format(yv.ndim, yv.shape))
#
# plt.plot(xv, yv, 'o--')
# plt.grid(True)
# plt.show()
# x = np.linspace(0, 500, 30)
# print("3" * 100)
# print(x.dtype)
# print("x的维度:{},shape:{}".format(x.ndim, x.shape))
# print(x)
# y = np.linspace(0, 500, 20)
# print("y的维度:{},shape:{}".format(y.ndim, y.shape))
# print(y)
#
# xv, yv = np.meshgrid(x, y)
# print("xv的维度:{},shape:{}".format(xv.ndim, xv.shape))
# print("yv的维度:{},shape:{}".format(yv.ndim, yv.shape))
#
# plt.plot(xv, yv, '.')
# plt.grid(True)
# plt.show()
# x = np.array([0, 0.5, 1])
# y = np.array([0,1])
#
# xv,yv = np.meshgrid(x, y)
# print("xv的维度:{},shape:{}".format(xv.ndim, xv.shape))
# print("yv的维度:{},shape:{}".format(yv.ndim, yv.shape))
# print(xv)
# print(yv)
#
# plt.plot(xv, yv, 'o--')
# plt.grid(True)
# plt.show()

x = np.array([0, 0.5, 1])
y = np.array([0,1])

xv,yv = np.meshgrid(x, y,indexing='ij')
print("xv的维度:{},shape:{}".format(xv.ndim, xv.shape))
print("yv的维度:{},shape:{}".format(yv.ndim, yv.shape))
print(xv)
print(yv)

plt.plot(xv, yv, 'o--')
plt.grid(True)
plt.show()

