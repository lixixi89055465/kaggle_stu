import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

data = pd.DataFrame([[1, 6, 7, 5, 1], [2, 10, 8, 3, 4], [3, 4, 0, 10, 2]],
                    columns=['val1', 'val2', 'val3', 'val4', 'val5'])
print(data)
sns.heatmap(data.corr(),linewidths=0.1,vmax=1.0, square=True,linecolor='white', annot=True)
plt.show()
