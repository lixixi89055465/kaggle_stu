import pandas as pd

pd.plotting.register_matplotlib_converters()
import matplotlib.pyplot as plt
import seaborn as sns

import os

if not os.path.exists("./input/museum_visitors.csv"):
    os.symlink("./input/input-for-datavis/museum_visitors.csv", "./input/museum_visitors.csv")
print("Setup Complete")
museum_filepath = "./input/museum_visitors.csv"
museum_data = pd.read_csv(museum_filepath, index_col='Date', parse_dates=True)
print(museum_data.head())
print(museum_data.dtypes)
print(museum_data.index)

print(museum_data.head(5))
print('2' * 100)
print(museum_data.shape)
print('3' * 100)
print(museum_data['2018-07']['Chinese American Museum'])
print('4' * 100)
print(museum_data['2018-10']['Avila Adobe'])
print(museum_data['2018-10']['Avila Adobe'] - museum_data['2018-10']['Firehouse Museum'])
print('5' * 100)
print(museum_data.columns)

# sns.lineplot(input=museum_data)
sns.lineplot(data=museum_data['Avila Adobe'])
plt.show()
