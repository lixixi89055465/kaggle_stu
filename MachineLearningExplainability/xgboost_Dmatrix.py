import xgboost as xgb
import pandas as pd  # 读取数据集
import numpy as np
import scipy

data = np.random.rand(5, 10)  # 5 entities, each contains 10 features
label = np.random.randint(2, size=5)  # binary target
dtrain = xgb.DMatrix(data, label=label)

# csr = scipy.sparse.csr_matrix((dat, (row, col)))
# dtrain = xgb.DMatrix(csr)
