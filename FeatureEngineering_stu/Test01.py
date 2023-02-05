import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from category_encoders import OrdinalEncoder
from category_encoders import OneHotEncoder

# # 准备数据
# df = pd.DataFrame({
#     "ID": [1, 2, 3, 4, 5, 6],
#     "RATING": ['G', 'B', 'G', 'B', 'B', 'G']
# })
# # 使用binary 编码的方式来编码类别变量
# encoder = category_encoders.BinaryEncoder(cols=['RATING']).fit(df)
# # 转换数据
# numeric_dataset = encoder.transform(df)
# print('1' * 100)
# print(numeric_dataset)
# 随机生成一些数据集

# train_set = pd.DataFrame(np.array([['male', 10], ['female', 20], ['male', 10],
#                                    ['female', 20], ['female', 15]]),
#                          columns=['Sex', 'Type'])
# train_y = np.array([False, True, True, False, False])
#
# print(train_set)
#
# # 随机生成一些测试集, 并有意让其包含未在训练集出现过的类别与缺失值
# test_set = pd.DataFrame(np.array([['female', 20], ['male', 20], ['others', 15],
#                                   ['male', 20], ['female', 40], ['male', 25]]),
#                         columns=['Sex', 'Type'])
# test_set.loc[4, 'Type'] = np.nan
# print("1" * 100)
# print(test_set)
# encoder = OrdinalEncoder(cols=['Sex', 'Type'],
#                          handle_unknown='value',
#                          handle_missing='value').fit(train_set, train_y)
# print('2' * 100)
# encoded_train = encoder.transform(train_set)
#
# print('3' * 100)
# print(encoded_train)
# print('4' * 100)
# print(test_set)
# encoded_test = encoder.transform(test_set)
# print(encoded_test)


# # 随机生成一些训练集
# train_set = pd.DataFrame(np.array([['male', 10], ['female', 20], ['male', 10],
#                                    ['female', 20], ['female', 15]]),
#                          columns=['Sex', 'Type'])
# train_y = np.array([False, True, True, False, False])
# # 随机生成一些测试集, 并有意让其包含未在训练集出现过的类别与缺失值
# test_set = pd.DataFrame(np.array([['female', 20], ['male', 20], ['others', 15],
#                                   ['male', 20], ['female', 40], ['male', 25]]),
#                         columns=['Sex', 'Type'])
# test_set.loc[4, 'Type'] = np.nan
#
# encoder = OneHotEncoder(cols=['Sex', 'Type'],
#                         handle_unknown='indicator',
#                         handle_missing='indicator',
#                         use_cat_names=True).fit(train_set, train_y)
# encoded_train = encoder.transform(train_set)  # 转换训练集
# encoded_test = encoder.transform(test_set)  # 转换测试集
# print(train_set.columns)
# print('2' * 100)
# print(encoded_train.columns)
# print('3' * 100)
# print(encoded_test)

# from category_encoders.target_encoder import TargetEncoder
#
# # 随机生成一些训练集
# train_set = pd.DataFrame(np.array([['male', 10], ['female', 20], ['male', 10],
#                                    ['female', 20], ['female', 15]]),
#                          columns=['Sex', 'Type'])
# train_y = np.array([False, True, True, False, False])
#
# # 随机生成一些测试集, 并有意让其包含未在训练集出现过的类别与缺失值
# test_set = pd.DataFrame(np.array([['female', 20], ['male', 20], ['others', 15],
#                                   ['male', 20], ['female', 40], ['male', 25]]),
#                         columns=['Sex', 'Type'])
# test_set.loc[4, 'Type'] = np.nan
#
# encoder = TargetEncoder(
#     cols=['Sex', 'Type'],
#     handle_missing='value',
#     handle_unknown='value',
# ).fit(train_set, train_y)
# print('1' * 100)
# print(train_set)
# print(encoder.transform(train_set))
# print('2' * 100)
# print(test_set)
# print(encoder.transform(test_set))
# # 验证一下计算的结果，在测试集中，‘male’类别的编码值为 0.473106
# prior = train_y.mean() # 先验概率
# min_samples_leaf = 1.0 # 默认为1.0
# smoothing = 1.0 # 默认为1.0
# n = 2 # 训练集中，两个样本包含‘male’这个标签


# import category_encoders as encoders
#
# X = pd.DataFrame({'col1': ['A', 'B', 'B', 'C', 'A']})
# y = pd.Series([1, 0, 1, 0, 1])
# enc = encoders.CatBoostEncoder()
# obtained= enc.fit_transform(X,y)
# print(obtained)

