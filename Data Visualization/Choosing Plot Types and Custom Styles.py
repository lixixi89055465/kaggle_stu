import pandas as pd

pd.plotting.register_matplotlib_converters()
import matplotlib.pyplot as plt
import seaborn as sns

print("Setup Complete")

# Path of the file to read
spotify_filepath = "./input/spotify.csv"

spotify_data = pd.read_csv(spotify_filepath, index_col='Date', parse_dates=True)
print(spotify_data.head())
print(spotify_data.columns)

plt.figure(figsize=(12, 6))
print('1' * 100)
# sns.lineplot(input=spotify_data)
print('2' * 100)
sns.set_style('darkgrid')
sns.lineplot(data=spotify_data)
plt.figure(figsize=(12, 6))
print('2' * 100)

plt.show()
