import pandas as pd

pd.plotting.register_matplotlib_converters()
import matplotlib.pyplot as plt
import seaborn as sns

print("Setup Complete")

cancer_filepath = "./input/cancer.csv"

# Fill in the line below to read the file into a variable cancer_data
cancer_data = pd.read_csv(cancer_filepath, index_col='Id')
print(cancer_data.head())
print(cancer_data.columns)

# Run the line below with no changes to check that you've loaded the data correctly

# Fill in the line below: In the first five rows of the data, what is the
# largest value for 'Perimeter (mean)'?
# max_perim = ____
print('1' * 100)
print(cancer_data.iloc[0:5]['Perimeter (mean)'].max())
# Fill in the line below: What is the value for 'Radius (mean)' for the tumor with Id 8510824?
# mean_radius = ____
# print(cancer_data['Radius (mean)'].max())
# print(cancer_data.loc[8510824])
print('2' * 100)
print(cancer_data.loc[8510824, 'Radius (mean)'])
# ['Radius (mean)']
print('3' * 100)
# sns.histplot(data=cancer_data, x='Area (mean)', hue='Diagnosis')
print('4' * 100)
sns.kdeplot(data=cancer_data, x='Radius (worst)', hue='Diagnosis', fill=True)
plt.show()
