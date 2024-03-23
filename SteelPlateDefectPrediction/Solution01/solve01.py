# -*- coding: utf-8 -*-
# @Time : 2024/3/23 下午4:57
# @Author : nanji
# @Site :
# @File : solve01.py
# @Software: PyCharm 
# @Comment : https://www.kaggle.com/code/getanmolgupta01/defect-pred-eda-xgboost-lgbm-catboost

'''

# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load
'''

import numpy as np  # linear algebra
import pandas as pd  # input processing, CSV file I/O (e.g. pd.read_csv)

# Input input files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory


import os
# for dirname, _, filenames in os.walk('../input/'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))


# Libray for Data Manipulation.
import pandas as pd
import numpy as np

# Library for Data Visualization.
import seaborn as sns
import matplotlib.pyplot as plt

sns.set(style="white", font_scale=1.5)
sns.set(rc={"axes.facecolor": "#FFFAF0", "figure.facecolor": "#FFFAF0"})
sns.set_context("poster", font_scale=.7)
import warnings

warnings.filterwarnings('ignore')

# Library to perform Statistical Analysis.
from scipy import stats
from scipy.stats import chi2
from scipy.stats import chi2_contingency

# Library to Display whole Dataset.
pd.set_option("display.max.columns", 100)

train_df = pd.read_csv('../input/train.csv')
test_df = pd.read_csv('../input/test.csv')
print(train_df.shape)
print(test_df.shape)
print(train_df.columns)

print("1" * 100)
print(train_df.info())

# Identify the input types of columns
column_data_types = train_df.dtypes
# print("2"*100)
print(column_data_types)

# Count the numerical and categorical columns
numerical_count = 0
categorical_count = 0

for column_name, data_type in column_data_types.items():
	if np.issubdtype(data_type, np.number):
		numerical_count += 1
	else:
		categorical_count += 1

# Print the counts
print(f"There are {numerical_count} Numerical Columns in dataset")
print(f"There are {categorical_count} Categorical Columns in dataset")

# 5. Checking if There's Any Duplicate Records.¶
print("Duplicates in Dataset: ", train_df.duplicated().sum())

# There are no duplicate records present in the dataset.
# 6. Computing Total No. of Missing Values and the Percentage of Missing Values¶
missing_data = train_df.isnull().sum().to_frame().rename(
	columns={0: "Total No. of Missing Values"}
)
missing_data['% of Missing Values']=round((
	missing_data['Total No. of Missing Values']/len(train_df))*100,2)
print('0' * 100)
print(missing_data)
# None of the Attribute are having Missing Values.
print('1'*100)
# print(round(train_df.describe().T, 2))
print(round(test_df.describe().T,2))
# The Minimum Pixels_Areas is 4 which conveys that some steel sheets are too small.
# 8. Dropping Attritbutes which doesn't imply any meaningful insights in our analysis.¶

