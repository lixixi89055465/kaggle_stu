import matplotlib.pyplot as plt
import numpy as np

from sklearn import datasets

# 本小节使用的是由 sklearn.datasets 中的 make_moons 函数生成的噪声为 0.25 的非线性虚拟数据集。使用非线性数据集是为了能够更好的看出决策树发生过拟合的样子，以及使用超参数解决过拟合后的结果
X, y = datasets.make_moons(noise=0.25, random_state=666)
# 通过散点图绘制数据集的分布。
plt.scatter(X[y == 0, 0], X[y == 0, 1])
plt.scatter(X[y == 1, 0], X[y == 1, 1])
plt.show()

from sklearn.tree import DecisionTreeClassifier

dt_clf = DecisionTreeClassifier()
dt_clf.fit(X, y)


# DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=None,
#                        max_features=None, max_leaf_nodes=None,
#                        min_impurity_decrease=0.0, min_impurity_split=None,
#                        min_samples_leaf=1, min_samples_split=2,
#                        min_weight_fraction_leaf=0.0, presort=False, random_state=None,
#                        splitter='best')


# 使用的 plot_decision_boundary 函数绘制决策边界。
def plot_decision_boundary(model, axis):
    x0, x1 = np.meshgrid(
        np.linspace(axis[0], axis[1], int((axis[1] - axis[0]) * 100)).reshape(-1, 1),
        np.linspace(axis[2], axis[3], int((axis[3] - axis[2]) * 100)).reshape(-1, 1),
    )
    X_new = np.c_[x0.ravel(), x1.ravel()]

    y_predict = model.predict(X_new)
    zz = y_predict.reshape(x0.shape)
    from matplotlib.colors import ListedColormap
    custom_cmap = ListedColormap(['#EF9A9A', '#FFF59D', '#90CAF9'])

    plt.contourf(x0, x1, zz, linewidth=5, cmap=custom_cmap)


plot_decision_boundary(dt_clf, axis=[-1.5, 2.5, -1.0, 1.5])
plt.scatter(X[y == 0, 0], X[y == 0, 1])
plt.scatter(X[y == 1, 0], X[y == 1, 1])
plt.show()
# 指定 max_depth 参数为 2，限制整个决策树的最大深度为 2
dt_clf2 = DecisionTreeClassifier(max_depth=2)
dt_clf2.fit(X, y)

plot_decision_boundary(dt_clf2, axis=[-1.5, 2.5, -1.0, 1.5])
plt.scatter(X[y == 0, 0], X[y == 0, 1])
plt.scatter(X[y == 1, 0], X[y == 1, 1])
plt.show()
# 指定 min_samples_split 参数为 10，节点再划分所需要的最小样本数为 10

dt_clf3 = DecisionTreeClassifier(min_samples_split=10)
dt_clf3.fit(X, y)

plot_decision_boundary(dt_clf3, axis=[-1.5, 2.5, -1.0, 1.5])
plt.scatter(X[y == 0, 0], X[y == 0, 1])
plt.scatter(X[y == 1, 0], X[y == 1, 1])
plt.show()
# 指定 min_samples_leaf 参数为 6，叶子节点最少样本数为 6

dt_clf4 = DecisionTreeClassifier(min_samples_split=6)
dt_clf4.fit(X, y)
plot_decision_boundary(dt_clf4, axis=[-1.5, 2.5, -1.0, 1.5])
plt.scatter(X[y == 0, 0], X[y == 0, 1])
plt.scatter(X[y == 1, 0], X[y == 1, 1])
plt.show()
# 指定 max_leaf_nodes 参数为 4，最大叶子节点数为 4
dt_clf5 = DecisionTreeClassifier(max_leaf_nodes=4)
dt_clf5.fit(X, y)
plot_decision_boundary(dt_clf5, axis=[-1.5, 2.5, -1.0, 1.5])
plt.scatter(X[y == 0, 0], X[y == 0, 1])
plt.scatter(X[y == 1, 0], X[y == 1, 1])
plt.show()
# 「决策树这种非参数学习很容易过拟合，所以在实际使用这些参数的时候，要注意避免决策树模型被过渡调节参数，从而导致决策树模型欠拟合。同时这些参数并不是相互独立的，它们之间可以相互组合，所以可以使用网格搜索的方式寻找最优的参数组合。」

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets

boston = datasets.load_boston()
X = boston.data
y = boston.target
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=666)
from sklearn.tree import DecisionTreeRegressor

dt_reg = DecisionTreeRegressor()
dt_reg.fit(X_train, y_train)

# DecisionTreeRegressor(criterion='mse', max_depth=None, max_features=None,
#                               max_leaf_nodes=None, min_impurity_decrease=0.0,
#                               min_impurity_split=None, min_samples_leaf=1,
#                               min_samples_split=2, min_weight_fraction_leaf=0.0,
#                               presort=False, random_state=None, splitter='best')
print(dt_reg.score(X_test, y_test))

print(dt_reg.score(X_train, y_train))