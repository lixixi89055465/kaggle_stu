import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import datetime

landslides = pd.read_csv('./data/catalog.csv')
np.random.seed(0)

# print(landslides.head())
# print(landslides.shape)
# print(landslides.columns)
# print(landslides.date.head())
print(landslides.dtypes)
print('1' * 100)
print(landslides['date'].dtype)
print(landslides.date.head())
landslides['date_parsed'] = pd.to_datetime(landslides['date'], format='%m/%d/%y')
print(landslides.dtypes)

print('2' * 100)
print(landslides['date_parsed'].head())

day_of_month_landslides = landslides['date_parsed'].dt.day
print(day_of_month_landslides.head())

# day_of_month_landslides = day_of_month_landslides.dropna()
# sns.displot(day_of_month_landslides, kde=False, bins=31)
# plt.show()
