import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

import shap

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

data = pd.read_csv('../input/hospital-readmissions/train.csv')

print(data.columns)
y = data.readmitted
base_features = ['number_inpatient', 'num_medications', 'number_diagnoses', 'num_lab_procedures',
                 'num_procedures', 'time_in_hospital', 'number_outpatient', 'number_emergency',
                 'gender_Female', 'payer_code_?', 'medical_specialty_?', 'diag_1_428', 'diag_1_414',
                 'diabetesMed_Yes', 'A1Cresult_None']
X = data[base_features].astype(float)
print(X.columns)
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=0)
small_val_X = val_X.iloc[:150]
my_model = RandomForestClassifier(n_estimators=30, random_state=1).fit(train_X, train_y)
print('0' * 100)

print(data.describe())
explainer = shap.TreeExplainer(my_model)
shap_values = explainer.shap_values(small_val_X)
shap.summary_plot(shap_values[1], small_val_X)

