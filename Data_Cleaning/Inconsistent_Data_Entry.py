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
print('0' * 100)
professors['Country'] = professors['Country'].str.lower()
professors['Country'] = professors['Country'].str.strip()

countries = professors['Country'].unique()
countries.sort()
print(countries)
print('1' * 100)

matches = fuzzywuzzy.process.extract('south korea',
                                     countries,
                                     limit=10,
                                     scorer=fuzzywuzzy.fuzz.token_sort_ratio)
print(matches)
print('2' * 100)


def replace_matcher_in_column(df, column, string_to_match, min_ratio=47):
    strings = df[column].unique()
    print(strings)
    matches = fuzzywuzzy.process.extract(string_to_match, strings,
                                         limit=10, scorer=fuzzywuzzy.fuzz.token_sort_ratio)
    # only get matcher with a ratio > 90
    close_matches = [matches[0] for matches in matches if matches[1] >= min_ratio]
    # get the rows of all the close matches in our dataframe
    rows_with_matches = df[column].isin(close_matches)
    # replace all rows with close matches with the input matches
    df.loc[rows_with_matches, column] = string_to_match
    print('All done!')



print('3' * 100)
replace_matcher_in_column(df=professors, column='Country', string_to_match='south korea')
