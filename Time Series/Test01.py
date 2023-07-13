import pandas as pd
import numpy as np
from sklearn.feature_selection import VarianceThreshold
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
import matplotlib.pyplot as plt
data = pd.read_csv(r"./train.csv")
x = data.iloc[:,1:]
y = data.iloc[:,0]
# print(x.shape)
selector = VarianceThreshold(np.median(x.var().values))
x_fsvar = selector.fit_transform(x)
score = []
for i in range(390,200,-10):
    x_fschi = SelectKBest(chi2,k = i).fit_transform(x_fsvar,y)
    once = cross_val_score(RFC(n_estimators=10,random_state=0),x_fschi,y,cv=5).mean()
    score.append(once)
plt.plot(range(390,200,-10),score)
plt.show()