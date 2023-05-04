from sklearn import linear_model
import numpy as np

reg = linear_model.RidgeCV(alphas=np.logspace(-6, 6, 13))
reg.fit([[0, 0], [0, 0], [1, 1]], [0, .1, .1])
print(reg.alpha_)
