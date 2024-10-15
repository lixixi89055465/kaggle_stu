# -*- coding: utf-8 -*-
# @Time : 2024/10/15 23:16
# @Author : nanji
# @Site : 
# @File : testH2OGridSearch.py
# @Software: PyCharm 
# @Comment :https://zhuanlan.zhihu.com/p/87107981
# Import H2O Grid Search:
import h2o
from h2o.automl import H2OAutoML

h2o.init()
# Import a sample binary outcome train/test set into H2O
train = h2o.import_file("https://s3.amazonaws.com/erin-data/higgs/higgs_train_10k.csv")
test = h2o.import_file("https://s3.amazonaws.com/erin-data/higgs/higgs_test_5k.csv")

# Identify predictors and response
x = train.columns
y = "response"
x.remove(y)

# For binary classification, response should be a factor
train[y] = train[y].asfactor()
test[y] = test[y].asfactor()

# Run AutoML for 30 seconds
aml = H2OAutoML(max_runtime_secs = 30)
aml.train(x = x, y = y,
          training_frame = train,
          leaderboard_frame = test)

# View the AutoML Leaderboard
lb = aml.leaderboard
lb
