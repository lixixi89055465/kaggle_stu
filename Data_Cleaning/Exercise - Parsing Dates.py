# modules we'll use
import pandas as pd
import numpy as np
import seaborn as sns
import datetime

# read in our input
earthquakes = pd.read_csv("./input/earthquake-database/database.csv")

# set seed for reproducibility
np.random.seed(0)

print(earthquakes.columns)
print(earthquakes.Date.dtype)
print('1' * 100)
date_lengths = earthquakes.Date.str.len()
print(date_lengths.value_counts())

print('2' * 100)
indices = np.where([date_lengths == 24])[1]
print(earthquakes.loc[indices])

print('3' * 100)
# print(earthquakes.Date.dtype)
for i in indices:
    earthquakes.loc[i, 'Date'] = pd.to_datetime(earthquakes.loc[i, 'Date'],
                                                infer_datetime_format=True).strftime('%m%d%y')
earthquakes['date_parsed']=pd.to_datetime(earthquakes['Date'],infer_datetime_format=True)

# print(earthquakes['date_parsed'].dt.day)
day_of_month_earthquakes = earthquakes['date_parsed'].dt.day
day_of_month_earthquakes = day_of_month_earthquakes.dropna()
sns.displot(day_of_month_earthquakes, kde=False, bins=31)