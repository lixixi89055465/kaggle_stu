import numpy as np
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn import preprocessing
from sklearn.manifold import TSNE

# 数据集导入
X, y = datasets.load_digits(return_X_y=True)

# t-SNE降维处理
# tsne = TSNE(n_components=8, verbose=1, random_state=42)
tsne = TSNE(n_components=8, verbose=1, )
result = tsne.fit_transform(X)

# 归一化处理
scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1))
result = scaler.fit_transform(result)
print(result.shape)

# 颜色设置
# color = ['#FFFAFA', '#BEBEBE', '#000080', '#87CEEB', '#006400',
#          '#00FF00', '#4682B4', '#D02090', '#8B7765', '#B03060']

# 可视化展示
# plt.figure(figsize=(10, 10))
# plt.title('t-SNE process')
# plt.xlim((-1.1, 1.1))
# plt.ylim((-1.1, 1.1))
# for i in range(len(result)):
#     plt.text(result[i,0], result[i,1], str(y[i]),
#              color=color[y[i]], fontdict={'weight': 'bold', 'size': 9})
# plt.scatter(result[:, 0], result[:, 1], c=y, s=10)
# plt.show()
