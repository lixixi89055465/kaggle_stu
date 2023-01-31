import pandas as pd
import numpy as np

from scipy import stats
from mlxtend.preprocessing import minmax_scaling
import seaborn as sns
import matplotlib.pyplot as plt

np.random.seed(0)

original_data = np.random.exponential(size=1000)
scaled_data = minmax_scaling(original_data, columns=[0])
# fig, ax = plt.subplots(1, 2, figsize=(15, 3))
# sns.histplot(original_data, ax=ax[0], kde=True, legend=False)
# ax[0].set_title("Original Data")
# sns.histplot(scaled_data, ax=ax[1], kde=True, legend=False)
# ax[1].set_title("Scaled input")
# plt.show()

normalized_data=stats.boxcox(original_data)

fig, ax = plt.subplots(1, 2, figsize=(15, 3))
sns.histplot(original_data, ax=ax[0], kde=True, legend=False)
ax[0].set_title("Original Data")
sns.histplot(normalized_data, ax=ax[1], kde=True, legend=False)
ax[1].set_title("normalized_data ")
plt.show()