import pandas as pd

df = pd.DataFrame({'consumption': [10.51, 103.11, 55.48],
                   'co2_emissions': [37.2, 19.66, 1712]},
                  index=['Pork', 'Wheat Products', 'Beef'])
print(df)

print('0' * 100)
print(df.idxmax(axis=1))
print('1' * 100)
print(df.idxmax(axis=0))
# importing pandas as pd
import pandas as pd

# Creating the dataframe
df = pd.DataFrame({"A": [4, 5, 2, None],
                   "B": [11, 2, None, 8],
                   "C": [1, 8, 66, 4]})

# Skipna = True will skip all the Na values
# find maximum along column axis
print('2' * 100)
print(df.idxmax(axis=1, skipna=True))
print('3' * 100)
import pandas as pd

countries = [
    'Afghanistan', 'Albania', 'Algeria', 'Angola',
    'Argentina', 'Armenia', 'Australia', 'Austria',
    'Azerbaijan', 'Bahamas', 'Bahrain', 'Bangladesh',
    'Barbados', 'Belarus', 'Belgium', 'Belize',
    'Benin', 'Bhutan', 'Bolivia', 'Bosnia and Herzegovina']
employment_values = [
    55.70000076, 51.40000153, 50.5, 75.69999695,
    58.40000153, 40.09999847, 61.5, 57.09999847,
    60.90000153, 66.59999847, 60.40000153, 68.09999847,
    66.90000153, 53.40000153, 48.59999847, 56.79999924,
    71.59999847, 58.40000153, 70.40000153, 41.20000076,
]
employment = pd.Series(employment_values, index=countries)
print(employment)
print('4' * 100)
max_country = employment.idxmax()
print(max_country)
max_country = employment.argmax()
print(max_country)
print('5'*100)
pure_employment = pd.Series(employment_values)
print(pure_employment)
print(pure_employment.idxmax())
print(pure_employment.argmax())
