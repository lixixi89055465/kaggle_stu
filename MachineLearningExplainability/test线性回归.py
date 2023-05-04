import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import datasets, linear_model

data = pd.read_csv('../input/CCPP/ccpp.csv')
X = data[['AT', 'V', 'AP', 'RH']]
y = data[['PE']]
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
from sklearn.linear_model import Ridge

ridge = Ridge(alpha=1)
ridge.fit(X_train, y_train)
print(ridge.coef_)
print(ridge.intercept_)
from sklearn.linear_model import RidgeCV

ridgecv = RidgeCV(alphas=[0.01, 0.1, 0.5, 1, 3, 7, 10, 20, 100])
ridgecv.fit(X_train, y_train)
print("0" * 100)
print(ridgecv.alpha_)
