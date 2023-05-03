import shap
import xgboost

shap.initjs()
X, y = shap.datasets.boston()
model = xgboost.train({'learning_rate': 0.01}, xgboost.DMatrix(X, label=y), 100)
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)
print("0" * 100)
print(shap_values.shape)
shap.force_plot(explainer.expected_value, shap_values[0, :], X.iloc[0, :])
print("1" * 100)
y_base = explainer.expected_value
print(y_base)
pred = model.predict(xgboost.DMatrix(X))
print(pred.mean())
