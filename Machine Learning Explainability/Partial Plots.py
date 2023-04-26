import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

data = pd.read_csv('../input/fifa-2018-match-statistics/FIFA 2018 Statistics.csv')
print(data.head())
print("0" * 100)
print(data.columns)
y = (data['Man of the Match'] == 'Yes')
print("1" * 100)
print(y.shape)
features_names = [i for i in data.columns if data[i].dtype in [np.int64]]
X = data[features_names]

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)
tree_model = DecisionTreeClassifier(random_state=0, max_depth=5, min_samples_split=5).fit(train_X, train_y)
from sklearn import tree
import graphviz

tree_graph = tree.export_graphviz(tree_model, out_file=None, feature_names=features_names)
graphviz.Source(tree_graph)

print("4" * 100)
from matplotlib import pyplot as plt
from sklearn.inspection import PartialDependenceDisplay

disp1 = PartialDependenceDisplay.from_estimator(tree_model, val_X, ['Goal Scored'])
plt.show()

feature_to_plot = 'Distance Covered (Kms)'
PartialDependenceDisplay.from_estimator(tree_model, val_X, [feature_to_plot])
plt.show()
print("5" * 100)
rf_model = RandomForestClassifier(random_state=0).fit(train_X, train_y)
disp3 = PartialDependenceDisplay.from_estimator(rf_model, val_X, [feature_to_plot])
plt.show()
fig, ax = plt.subplots(figsize=(8, 6))
f_names = [('Goal Scored', 'Distance Covered (Kms)')]
disp4 = PartialDependenceDisplay.from_estimator(tree_model, val_X, f_names, ax=ax)
plt.show()
