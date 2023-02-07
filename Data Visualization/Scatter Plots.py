import pandas as pd

pd.plotting.register_matplotlib_converters()
import matplotlib.pyplot as plt
import seaborn as sns

print('Setup Complete')
# Set up code checking
import os

candy_filepath = "./candy.csv"
candy_data = pd.read_csv(candy_filepath, index_col='id')
print('1' * 100)
print(candy_data.head(5))
print(candy_data.shape)
print(candy_data.columns)
print(candy_data.index)

print('3' * 100)
print(candy_data[['competitorname', 'winpercent']])
# sns.scatterplot(x=candy_data['sugarpercent'], y=candy_data['winpercent'])
# sns.regplot(x=candy_data['sugarpercent'], y=candy_data['winpercent'])

# sns.scatterplot(x=candy_data['pricepercent'], y=candy_data['winpercent'], hue=candy_data['chocolate'])
# sns.lmplot(
#     x='sugarpercent', y='winpercent',
#     hue='chocolate',
#     data=candy_data
# )

# sns.catplot(candy_data, x='chocolate', y='winpercent')
sns.swarmplot(x=candy_data['chocolate'], y=candy_data['winpercent'])

plt.show()
