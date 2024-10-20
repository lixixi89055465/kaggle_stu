# -*- coding: utf-8 -*-
# @Time : 2024/10/20 18:11
# @Author : nanji
# @Site : https://zhuanlan.zhihu.com/p/540956200
# @File : testcatboost01.py
# @Software: PyCharm 
# @Comment :
from sklearn.datasets import load_boston
import catboost
from catboost import CatBoostRegressor

boston = load_boston()
y = boston['target']
X = boston['data']
pool = catboost.Pool(data=X, label=y)
model = CatBoostRegressor(depth=2, verbose=False, iterations=1).fit(X, y)
model.plot_tree(tree_idx=0, )
# pool=pool

from catboost import CatBoostClassifier, Pool, metrics, cv

from catboost.datasets import titanic

titanic_df = titanic()
X = titanic_df[0].drop('Survived', axis=1)
y = titanic_df[0].Survived
from catboost import CatBoostClassifier
from multiprocessing import Pool

# pool = Pool(X, y, cat_features=cat_features_index, feature_names=list(X.columns))
# # 分类变量的缺失值用"NAN"填充，代码略
# model = CatBoostClassifier(
#     max_depth=2, verbose=False, max_ctr_complexity=1, random_seed=42, iterations=2).fit(pool)
# model.plot_tree(
#     tree_idx=0,
#     pool=pool # 对于一个需要使用独热编码的特征，"pool" 是一个必须的参数
# )

from catboost.datasets import titanic
import numpy as np

train_df, test_df = titanic()
# 用一些超出分布范围的数字来填充缺失值
train_df.fillna(-999, inplace=True)
test_df.fillna(-999, inplace=True)
# 拆分特征变量及标签变量
X = train_df.drop('Survived', axis=1)
y = train_df.Survived
# 划分训练集和测试集
from sklearn.model_selection import train_test_split

X_train, X_validation, y_train, y_validation = train_test_split(X, y, train_size=0.75, random_state=42)
X_test = test_df

from catboost import CatBoostClassifier, Pool, metrics, cv
from sklearn.metrics import accuracy_score

model = CatBoostClassifier(
    custom_loss=[metrics.Accuracy()],  # 该指标可以计算logloss，并且在该规模的数据集上更加光滑
    random_seed=42,
    logging_level='Silent'
)

categorical_features_indices = np.where(X.dtypes != np.float64)[0]

# 模型训练
model.fit(
    X_train, y_train,
    cat_features=categorical_features_indices,
    eval_set=(X_validation, y_validation),
    #     logging_level='Verbose',  # you can uncomment this for text output
    plot=True
);

# 特征变量统计
# Float feature
# feature = 'Fare'
# res = model.calc_feature_statistics(
#       X_train, y_train, feature, plot=True)
#
print('3' * 100)

import pandas as pd
import os
import numpy as np

np.set_printoptions(precision=4)
import catboost
from catboost import *
from catboost import datasets

(train_df, test_df) = catboost.datasets.amazon()
print(train_df.head())
y = train_df.ACTION
X = train_df.drop('ACTION', axis=1)
print('Label :{}'.format(set(y)))
print('Zero count = {}, One count = {}'.format(len(y) - sum(y), sum(y)))

dataset_dir = './amazon'
if not os.path.exists(dataset_dir):
    os.makedirs(dataset_dir)

train_df.to_csv(
    os.path.join(dataset_dir, 'train.csv'),
    index=False, sep=',', header=True
)

test_df.to_csv(
    os.path.join(dataset_dir, 'test.csv'),
    index=False, sep=',', header=True
)
print('5'*100)
from catboost.utils import create_cd
feature_names = dict()
for column, name in enumerate(train_df):
    if column == 0:
        continue
    feature_names[column - 1] = name
create_cd(
    label=0,
    cat_features=list(range(1, train_df.columns.shape[0])),
    feature_names=feature_names,
    output_path=os.path.join(dataset_dir, 'train.cd')
)
# cat_features=['RESOURCE', 'MGR_ID', 'ROLE_ROLLUP_1', 'ROLE_ROLLUP_2', 'ROLE_DEPTNAME',
#        'ROLE_TITLE', 'ROLE_FAMILY_DESC', 'ROLE_FAMILY', 'ROLE_CODE']
cat_features= feature_names
pool1 = Pool(data=X, label=y, cat_features=cat_features)
pool2 = Pool(
    data=os.path.join(dataset_dir, 'train.csv'),
    delimiter=',',
    column_description=os.path.join(dataset_dir, 'train.cd'),
    has_header=True
)
pool3 = Pool(data=X, cat_features=cat_features)
X_prepared = X.values.astype(str).astype(object)
# 对于FeaturesData类，类别特性必须具有str类型
pool4 = Pool(
    data=FeaturesData(
        cat_feature_data=X_prepared,
        cat_feature_names=list(X)
    ),
    label=y.values
)

print('Dataset shape')
print('dataset 1:' + str(pool1.shape) +
      '\ndataset 2:' + str(pool2.shape) +
      '\ndataset 3:' + str(pool3.shape) +
      '\ndataset 4:' + str(pool4.shape))
print('\n')
print('Column names')
print('dataset 1:')
print(pool1.get_feature_names())
print('\ndataset 2:')
print(pool2.get_feature_names())
print('\ndataset 3:')
print(pool3.get_feature_names())
print('\ndataset 4:')
print(pool4.get_feature_names())

from sklearn.model_selection import train_test_split
X_train, X_validation, y_train, y_validation = train_test_split(
     X, y, train_size=0.8, random_state=1234)

from catboost import CatBoostClassifier
model = CatBoostClassifier(
    iterations=5,
    learning_rate=0.1,
    # loss_function='CrossEntropy'
)