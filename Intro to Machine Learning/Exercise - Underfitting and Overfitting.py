import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

# Path of the file to read
iowa_file_path = '../input/home-input-for-ml-course/train.csv'

home_data = pd.read_csv(iowa_file_path)

print(home_data.head())
print(home_data.columns)
# Create target object and call it y
y = home_data.SalePrice
# Create X
features = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']
X = home_data[features]
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)
# Split into validating and training input
# Specify Model
iowa_model = DecisionTreeRegressor(random_state=1)
# Fit model
iowa_model.fit(train_X, train_y)
val_predictions = iowa_model.predict(val_X)
val_mae = mean_absolute_error(val_predictions, val_y)
print('Validation MAX:{:,.0f}'.format(val_mae))


# Set up code checking

def get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(train_X, train_y)
    preds_val = model.predict(val_X)
    mae = mean_absolute_error(val_y, preds_val)
    return (mae)


candidate_max_leaf_nodes = [5, 25, 50, 100, 250, 500]
# Write loop to find the ideal tree size from candidate_max_leaf_nodes
best_tree_size = 0

result = {
    max_leaf_nodes: get_mae(max_leaf_nodes=max_leaf_nodes, train_X=train_X, val_X=val_X, train_y=train_y, val_y=val_y)
    for
    max_leaf_nodes in candidate_max_leaf_nodes}

best_tree_size = min(result, key=result.get)
print(best_tree_size)

final_model = DecisionTreeRegressor(max_leaf_nodes=best_tree_size,random_state=1)
final_model.fit(X, y)
