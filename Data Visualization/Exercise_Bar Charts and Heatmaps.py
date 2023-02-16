import pandas as pd

pd.plotting.register_matplotlib_converters()
import matplotlib.pyplot as plt
import seaborn as sns

print("Setup Complete")
# Set up code checking
import os

if not os.path.exists("./input/ign_scores.csv"):
    os.symlink("./input/data-for-datavis/ign_scores.csv", "./input/ign_scores.csv")

# Path of the file to read
ign_filepath = "./input/ign_scores.csv"
ign_data = pd.read_csv(ign_filepath, index_col='Platform')
print(ign_data.head())
print(ign_data.columns)
print('2' * 100)
print(ign_data.sum(axis=1) / ign_data.shape[0])
# print((ign_data.sum(axis=0) / ign_data.shape[1]))
# print((ign_data.sum(axis=1) / ign_data.shape[0]).max())
print('3' * 100)
# print(ign_data.max())
plt.figure(figsize=(8, 6))
sns.barplot(x=ign_data['Racing'], y=ign_data.index)
plt.show()
print('4' * 100)
print(ign_data.shape)
print('5' * 100)
sns.heatmap(ign_data,annot=True)
plt.xlabel('Genre')

plt.show()
