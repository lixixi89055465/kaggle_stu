import pandas as pd
from sklearn.tree import DecisionTreeRegressor

# Path of the file to read
iowa_file_path = '../input/home-input-for-ml-course/train.csv'
home_data = pd.read_csv(iowa_file_path)
print(home_data.head())
print(home_data.columns)
feature_columns = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']
X = home_data[feature_columns]
y = home_data.SalePrice
iowa_model = DecisionTreeRegressor(random_state=0)
iowa_model.fit(X, y)
print(iowa_model.predict(X.head()))
print("Actual target values for those homes:", y.head().tolist())
from sklearn.model_selection import train_test_split

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)

# You imported DecisionTreeRegressor in your last exercise
# and that code has been copied to the setup code above. So, no need to
# import it again

# Specify the model
iowa_model = DecisionTreeRegressor(random_state=1)

# Fit iowa_model with the training input.
iowa_model.fit(train_X, train_y)

val_predictions = iowa_model.predict(val_X)
print(val_predictions[:5])
print(val_y[:5])
from sklearn.metrics import mean_absolute_error

val_mae = mean_absolute_error(y_pred=val_predictions, y_true=val_y)
print(val_mae)
