# modules we'll use
import pandas as pd
import numpy as np

# read in all our input
sf_permits = pd.read_csv("./input/Building_Permits.csv")

# set seed for reproducibility
np.random.seed(0)
# print(sf_permits.head())
print(sf_permits.isna().sum().sum() / np.product(sf_permits.shape) * 100)
print(sf_permits.columns)
features = ['Street Number Suffix', 'Zipcode']
print(sf_permits[features].head())
print(sf_permits.dropna().shape)
sf_permits_with_na_droped = sf_permits.dropna(axis=1).shape
dropped_columns=sf_permits.shape[1]-sf_permits_with_na_droped[1]
print(dropped_columns)
sf_permits_with_na_imputed = sf_permits.bfill(axis=0).fillna(0)
