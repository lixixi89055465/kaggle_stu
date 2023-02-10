# Code you have previously used to load data
import pandas as pd
from sklearn.tree import DecisionTreeRegressor

# Path of the file to read
iowa_file_path = '../input/home-data-for-ml-course/train.csv'

home_data = pd.read_csv(iowa_file_path)
print(home_data.head())
print(home_data.columns)
a = home_data['SalePrice']
print(a.head())
feature_names = ["LotArea", "YearBuilt", "1stFlrSF", "2ndFlrSF", "FullBath", "BedroomAbvGr", "TotRmsAbvGrd"]
X = home_data[feature_names]
y = home_data.SalePrice
print('1' * 100)
print(X.describe())
iowa_model = DecisionTreeRegressor(random_state=1)
iowa_model.fit(X, y)

print('2' * 100)
predictions = iowa_model.predict(X)
print(predictions)
