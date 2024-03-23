'''

'''

import pandas as pd

pd.plotting.register_matplotlib_converters()
import matplotlib.pyplot as plt
import seaborn as sns

print("Setup Complete")
# Path of the file to read
iris_filepath = "./input/iris.csv"

# Read the file into a variable iris_data
iris_data = pd.read_csv(iris_filepath, index_col="Id")

# Print the first 5 rows of the input
print(iris_data.head())
print(iris_data.columns)

print('2' * 100)
# sns.histplot(iris_data['Petal Width (cm)'])
print('3' * 100)
# sns.kdeplot(input=iris_data['Petal Length (cm)'], shade=True)
print('4' * 100)
# sns.jointplot(x=iris_data['Petal Length (cm)'], y=iris_data['Sepal Width (cm)'], kind='kde')
print('5' * 100)
# sns.jointplot(x=iris_data['Petal Length (cm)'], y=iris_data['Sepal Width (cm)'], kind='scatter')
print('6' * 100)
# sns.histplot(input=iris_data, x='Petal Length (cm)', hue='Species')
print('7' * 100)
# sns.kdeplot(input=iris_data, x='Petal Length (cm)', hue='Species', shade=True)
sns.kdeplot(data=iris_data, x='Petal Length (cm)', hue='Species', fill=True)
# Add title
plt.title("Distribution of Petal Lengths, by Species")
plt.show()
