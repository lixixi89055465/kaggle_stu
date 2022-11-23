import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

plt.style.use("seaborn-whitegrid")
plt.rc("figure", autolayout=True)
plt.rc(
    "axes",
    labelweight="bold",
    labelsize="large",
    titleweight="bold",
    titlesize=14,
    titlepad=10,
)
accidents = pd.read_csv("./input/fe-course-data/accidents.csv")
autos = pd.read_csv("./input/fe-course-data/autos.csv")
concrete = pd.read_csv("./input/fe-course-data/concrete.csv")
customer = pd.read_csv("./input/fe-course-data/customer.csv")

print(autos.columns)

autos['stroke_ratio']=autos.stroke/autos.bore
print(autos[['stroke', 'bore', 'stroke_ratio']].head())
autos["displacement"] = (
    np.pi * ((0.5 * autos.bore) ** 2) * autos.stroke * autos.num_of_cylinders
)
print('1'*100)
accidents['LogWindSpeed']=accidents.WindSpeed.apply(np.log1p)
print(accidents['LogWindSpeed'].head())
# plot a comparison

fig,axs=plt.subplots(1,2,figsize=(8,4))
sns.kdeplot(accidents.WindSpeed,shade=True,ax=axs[0])
sns.kdeplot(accidents.LogWindSpeed, shade=True, ax=axs[1]);

