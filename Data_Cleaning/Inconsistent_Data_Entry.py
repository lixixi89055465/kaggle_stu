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
print('0'*100)
professors['Country'] = professors['Country'].str.lower()
professors['Country'] = professors['Country'].str.strip()

countries = professors['Country'].unique()
countries.sort()
print(countries)
print('1'*100)

matches = fuzzywuzzy.process.extract('south korea',
                                     countries,
                                     limit=10,
                                     scorer=fuzzywuzzy.fuzz.token_sort_ratio)
print(matches)
print('2'*100)

