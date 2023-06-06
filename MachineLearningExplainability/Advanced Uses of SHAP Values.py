import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

data = pd.read_csv('../input/Predict FIFA 2018 Man of the Match/FIFA 2018 Statistics.csv')

y = (data['Man of the Match'] == 'Yes')
feature_name = [i for i in data.columns if data[i].dtype in [np.int64]]
print(feature_name)
X = data[feature_name]
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)
my_model = RandomForestClassifier(random_state=0).fit(train_X, train_y)

import shap

explainer = shap.TreeExplainer(my_model)
shap_values = explainer.shap_values(val_X)
shap.summary_plot(shap_values[1], val_X)
