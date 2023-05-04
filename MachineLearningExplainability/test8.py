from sklearn import linear_model
import numpy as np

reg = linear_model.Lasso(alpha=0.1)

reg.fit([[0, 0], [1, 1]], [0, 1])
print(reg.predict(([[1, 1, ]])))
