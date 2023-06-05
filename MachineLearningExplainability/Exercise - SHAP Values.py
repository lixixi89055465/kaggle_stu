import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

data = pd.read_csv('../input/hospital-readmissions/train.csv')
print("0" * 100)
print(data.columns)
y = data.readmitted
base_features = [c for c in data.columns if c != 'readmitted']
X = data[base_features]

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)
my_model = RandomForestClassifier(n_estimators=30, random_state=1).fit(train_X, train_y)
import eli5

print("1" * 100)
from eli5.sklearn import PermutationImportance

perm = PermutationImportance(my_model, random_state=1).fit(val_X, val_y)
eli5.show_weights(perm, feature_names=val_X.columns.tolist())

print("2" * 100)
from matplotlib import pyplot as plt
from sklearn.inspection import PartialDependenceDisplay

feature_name = 'number_inpatient'
PartialDependenceDisplay.from_estimator(my_model, val_X, [feature_name])
plt.show()
print("3" * 100)
fig, ax = plt.subplots(figsize=(8, 6))
f_names = [('number_inpatient', 'time_in_hospital')]
disp4 = PartialDependenceDisplay.from_estimator(my_model, val_X, f_names, ax=ax)
plt.show()
print("4" * 100)
all_train = pd.concat([train_X, train_y], axis=1)
all_train.groupby(['time_in_hospital']).mean().readmitted.plot()

# plt.show()

print("5" * 100)
row_to_show = 5
data_for_prediction = val_X.iloc[row_to_show]
data_for_prediction_array = data_for_prediction.values.reshape(1, -1)
my_model.predict_proba(data_for_prediction_array)
import shap

explainer = shap.TreeExplainer(my_model)
shap_values = explainer.shap_values(data_for_prediction)
shap.initjs()
shap.force_plot(explainer.expected_value[1], shap_values[1], data_for_prediction)

import shap

sample_data_for_prediction = val_X.iloc[0].astype(float)


def _patient_risk_factors(model, patient_data):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(patient_data)
    shap.initjs()
    return shap.force_plot(explainer.expected_value[1], shap_values[1], patient_data)


_patient_risk_factors(model=my_model, patient_data=sample_data_for_prediction)
