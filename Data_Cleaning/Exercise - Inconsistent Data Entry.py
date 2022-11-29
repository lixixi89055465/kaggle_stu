import pandas as pd
import numpy as np
import fuzzywuzzy
from fuzzywuzzy import process
import chardet

professors = pd.read_csv('./pakistan-intellectual-capital/pakistan_intellectual_capital.csv')

np.random.seed(0)

professors['Country'] = professors['Country'].str.lower()
professors['Country'] = professors['Country'].str.strip()

# get the top 10 closest matches to "south korea"
countries = professors['Country'].unique()
matches = fuzzywuzzy.process.extract("south korea", countries, limit=10, scorer=fuzzywuzzy.fuzz.token_sort_ratio)


def replace_matches_in_column(df, column, string_to_match, min_ratio=47):
    strings = df[column].unique()
    matches = fuzzywuzzy.process.extract(string_to_match, strings,
                                         limit=10, scorer=fuzzywuzzy.fuzz.token_sort_ratio)
    close_matches = [matches[0] for matches in matches if matches[1] >= min_ratio]
    print(close_matches)
    rows_with_matches = df[column].isin(close_matches)
    df.loc[rows_with_matches, column] = string_to_match

    # let us know the function's done
    print("All done!")


# replace_matches_in_column(df=professors, column='Country', string_to_match="south korea")

countries = professors['Country'].unique()

a=professors['Graduated from']=professors['Graduated from'].str.strip()

row_with_matches=professors['Country'].isin(['usofa'])
print(row_with_matches)
professors.loc[row_with_matches,'Country']='usa'
print('2'*100)
print(professors['Country'].unique())


