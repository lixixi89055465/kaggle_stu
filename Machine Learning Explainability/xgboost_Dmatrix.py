import xgboost as xgb
import pandas as pd  # 读取数据集

data = pd.read_csv('data.csv')  # 将数据集转换为XGBoost中的内部数据格式
dtrain = xgb.DMatrix(data=data.iloc[:, :-1], label=data.iloc[:, -1])
