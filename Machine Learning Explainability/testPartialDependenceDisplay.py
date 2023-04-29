print(__doc__)

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.tree import DecisionTreeRegressor
from sklearn.inspection import plot_partial_dependence

diabetes = load_diabetes()
X = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
y = diabetes.target

tree = DecisionTreeRegressor()
mlp = make_pipeline(StandardScaler(),
                    MLPRegressor(hidden_layer_sizes=(100, 100),
                                 tol=1e-3, max_iter=500, random_state=0))
tree.fit(X, y)
mlp.fit(X, y)
fig, ax = plt.subplots(figsize=(12, 6))
ax.set_title('Decision Tree')
tree_disp = plot_partial_dependence(tree, X, ['age', 'bmi'], ax=ax)
plt.show()
print("2" * 100)
fig, ax = plt.subplots(figsize=(12, 6))
ax.set_title('Multi-layer Perception')
mlp_disp = plot_partial_dependence(mlp, X, ['age', 'bmi'], ax=ax)
plt.show()
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
tree_disp.plot(ax=ax1)
ax1.set_title("Decision Tree")
mlp_disp.plot(ax=ax2)
ax2.set_title("Multi-layer Perceptron")
plt.show()
print("5"*100)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 6))
tree_disp.plot(ax=[ax1, ax2], line_kw={"label": "Decision Tree"})
mlp_disp.plot(ax=[ax1, ax2], line_kw={"label": "Multi-layer Perceptron",
                                      "color": "red"})
ax1.legend()
ax2.legend()
plt.show()
