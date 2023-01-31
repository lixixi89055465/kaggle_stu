# modules we'll use
import pandas as pd
import numpy as np

# read in all our input
nfl_data = pd.read_csv("./input/NFL Play by Play 2009-2017 (v4).csv")

# set seed for reproducibility
np.random.seed(0)
# print(nfl_data.head())
missing_values_count = nfl_data.isnull().sum()
# print(missing_values_count)
# print(nfl_data.shape)
# how many total missing values do we have
total_cells = np.product(nfl_data.shape)
total_missings = missing_values_count.sum()

# percent of input that is missing
percent_missing = (total_missings / total_cells) * 100
# print(percent_missing)

columns_with_na_dropped = nfl_data.dropna(axis=1)
# print(columns_with_na_dropped.head())
subset_nfl_data = nfl_data.loc[:, 'EPA':'Season'].head()
# print(subset_nfl_data)

# print("Columns in original dataset:%d \n" % nfl_data.shape[1])
# print("Columns with na's dropeed :%d \n" % columns_with_na_dropped.shape[1])

print('1'*100)
# print(subset_nfl_data.fillna(0))
print(subset_nfl_data.fillna(method='bfill', axis=0).fillna(0))

