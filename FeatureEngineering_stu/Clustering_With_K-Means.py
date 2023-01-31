import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans

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

df = pd.read_csv("./input/fe-course-input/housing.csv")
print("1" * 100)
print(df.head())
print(df.columns)
X = df.loc[:, ["MedInc", "Latitude", "Longitude"]]
kmeans = KMeans(n_clusters=6)
X['Cluster'] = kmeans.fit_predict(X)
X['Cluster'] = X['Cluster'].astype('category')
print(X[['Longitude', 'Latitude', 'Cluster']].head())
sns.relplot(
    x="Longitude", y="Latitude", hue="Cluster", data=X, height=6,
)

print("1" * 100)
X["MedHouseVal"] = df["MedHouseVal"]

# sns.catplot(x='MedHouseVal', y='Cluster', input=X,
#             kind='boxen', height=6)
sns.catplot(x='MedHouseVal', y='Cluster', data=X,
            kind='box', height=6)
plt.show()
