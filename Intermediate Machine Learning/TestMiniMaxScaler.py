import numpy as np

from sklearn.preprocessing import MinMaxScaler

data = [[-1, 2], [-0.5, 6], [0, 10], [1, 18]]

import pandas as pd

pd.DataFrame(data)

# 实现归一化
scaler = MinMaxScaler()  # 实例化
scaler.fit(data)  # fit，在这里本质是生成min(x)和max(x)
result = scaler.transform(data)  # 通过接口导出结果
print(result)
print("0" * 100)
result_ = scaler.fit_transform(data)
print(scaler.inverse_transform(result))

print("1"*100)
scaler=MinMaxScaler(feature_range=[5,10])
result=scaler.fit_transform(data)
print(result)

