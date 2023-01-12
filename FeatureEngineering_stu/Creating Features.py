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
accidents = pd.read_csv("./data/fe-course-data/accidents.csv")
autos = pd.read_csv("./data/fe-course-data/autos.csv")
concrete = pd.read_csv("./data/fe-course-data/concrete.csv")
customer = pd.read_csv("./data/fe-course-data/customer.csv")

# print(autos.columns)
#
# autos['stroke_ratio'] = autos.stroke / autos.bore
# print(autos[['stroke', 'bore', 'stroke_ratio']].head())
# autos["displacement"] = (
#         np.pi * ((0.5 * autos.bore) ** 2) * autos.stroke * autos.num_of_cylinders
# )
# print('1' * 100)
# accidents['LogWindSpeed'] = accidents.WindSpeed.apply(np.log1p)
# print(accidents['WindSpeed'].head())
# print(accidents['LogWindSpeed'].head())
# # plot a comparison
#
# # fig,axs=plt.subplots(1,2,figsize=(8,4))
# # sns.kdeplot(accidents.WindSpeed,shade=True,ax=axs[0])
# # sns.kdeplot(accidents.LogWindSpeed, shade=True, ax=axs[1]);
# roadway_features = ["Amenity", "Bump", "Crossing", "GiveWay",
#                     "Junction", "NoExit", "Railway", "Roundabout", "Station", "Stop",
#                     "TrafficCalming", "TrafficSignal"]
# accidents['RoadwayFeature'] = accidents[roadway_features].sum(axis=1)
# components = ["Cement", "BlastFurnaceSlag", "FlyAsh", "Water",
#               "Superplasticizer", "CoarseAggregate", "FineAggregate"]
# concrete['Components'] = concrete[components].gt(0).sum(axis=1)
# print('3' * 100)
# print(concrete[components].head(10))
# print(concrete['Components'].head(10))
# print('4' * 100)
# print(customer['Policy'].head(10))
# print(customer['Policy'].head(10).str.split(" ", expand=True))
# customer[['Type', 'Level']] = customer['Policy'].head(10).str.split(" ", expand=True)
# print('5"*100')
# print(customer[['Policy', 'Type', 'Level']].head(10))

# print('6' * 100)
# autos['make_and_style'] = autos['make'] + "_" + autos['body_style']
# print(autos['make_and_style'].head(10))
#
# print("7" * 100)
# customer['AverageIncome'] = customer.groupby("State")['Income'].transform("mean")
# print("8" * 100)
# print(customer[['State', 'Income', 'AverageIncome']].head(10))
# customer['StateFreq'] = customer.groupby('State')['State'].transform('count') / customer.State.count()
# print("9" * 100)
# print(customer[['State', 'StateFreq']].head(10))
print("9" * 100)
df_train = customer.sample(frac=0.5)
df_valid = customer.drop(df_train.index)
df_train['AverageClaim'] = df_train.groupby('Coverage')['ClaimAmount'].transform('mean')
print(df_train.shape)
print(df_valid.shape)
df_valid = df_valid.merge(
    df_train[['Coverage', 'AverageClaim']].drop_duplicates(),
    on='Coverage',
    how='left'
)
print("1"*100)
print(df_train.shape)
print(df_valid.shape)
