import numpy as np
from sklearn.impute import SimpleImputer

imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
x = [[7, 2, 3],
     [4, np.nan, 6],
     [10, 3, 9]]
data = [[np.nan, 2, 3],
        [4, np.nan, 6],
        [10, np.nan, 9]]

Fit = imp_mean.fit(x)
new_data = imp_mean.transform(data)

print("1" * 100)
print(x)
print("2" * 100)
print(Fit)
print("3" * 100)
print(data)
print("4" * 100)
print(new_data)
print("5" * 100)
X1 = np.array([[1, 2, np.nan],
               [4, np.nan, 6],
               [np.nan, 8, 9]])
imp = SimpleImputer(missing_values=np.nan, strategy='mean')
print(imp.fit_transform(X1))
print("6"*100)
print(imp.get_params())
print("7"*100)