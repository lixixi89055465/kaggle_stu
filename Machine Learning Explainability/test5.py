from sklearn.preprocessing import StandardScaler
import numpy as np

X = np.array([[1., -1., 2.],
              [2., 0., 0.],
              [0., 1., -1.],
              ])
from sklearn import preprocessing

X_scaled = preprocessing.scale(X)
print(X_scaled)

s1=StandardScaler()
s1.fit(X)
print("1"*100)
print(s1.transform(X))
