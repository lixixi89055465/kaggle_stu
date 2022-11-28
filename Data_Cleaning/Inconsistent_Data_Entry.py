'''

'''
import pandas as pd
import numpy as np
import fuzzywuzzy
from fuzzywuzzy import process
import chardet

professors = pd.read_csv("./pakistan-intellectual-capital/pakistan_intellectual_capital.csv")
np.random.seed(0)

print(professors.head())

countries = professors['Country'].unique()
countries.sort()
print(countries)
professors['Country'] = professors['Country'].str.lower()
professors['Country'] = professors['Country'].str.strip()

countries=professors['Country'].unique()
countries.sort()
print(countries)