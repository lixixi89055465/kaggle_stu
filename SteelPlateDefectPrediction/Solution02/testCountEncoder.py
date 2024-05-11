# -*- coding: utf-8 -*-
# @Time    : 2024/5/9 下午10:44
# @Author  : nanji
# @Site    : https://www.cnblogs.com/dangui/p/15836197.html#4%E8%AE%A1%E6%95%B0%E7%BC%96%E7%A0%81countencoder
# @File    : testCountEncoder.py
# @Software: PyCharm 
# @Comment :


import category_encoders as ce
import pandas as pd
from statsmodels.formula.api import ols

data = pd.DataFrame({'ID': [1, 2, 3, 4, 5, 6, 7, 8], \
					 'Sex': ['F', 'M', 'M', 'F', 'M', None, 'F', 'M'], \
					 'BloodType': ['A', 'AB', 'O', 'B', None, 'O', 'AB', 'B'], \
					 'Grade': ['High', None, 'Medium', 'Low', 'Low', 'Medium', 'Low', 'High'], \
					 # 'Height':[156, None, 167, 175, 164, 180], 'Weight':[50, None, 65, 67, 48, 76],
					 'Education': ['PhD', 'HighSchool', 'Bachelor', 'Master', 'HighSchool', 'Master', 'PhD',
								   'Bachelor'], \
					 'Income': [28300, 4500, 7500, 12500, 4200, 15000, 25000, 7200]})

print('\nOriginal Dataset:\n', data)

# ce_1 = ce.OrdinalEncoder(cols=['Grade', 'Education'],
# mapping = [
# 	{
# 		'col': 'Grade', \
# 		'mapping': {None: 0, 'Low': 1, 'Medium': 2, 'High': 3} \
# 		}, \
# 	{
# 		'col': 'Education', \
# 		'mapping': {None: 0, 'HighSchool': 1, 'Bachelor': 2, 'Master': 3, 'PhD': 4} \
# 		}
# ]).fit_transform(data)
# print('\nOrdinalEncoder Return the transformed dataset:\n', ce_1)

# ce_2 = ce.OneHotEncoder(cols=['BloodType'], use_cat_names=True).fit_transform(data)
# ce_2 = ce.OneHotEncoder(cols=['BloodType'], use_cat_names=False).fit_transform(data)
# print('1'*100)
# print('\n OneHotEncoder Return the transformed dataset :\n', ce_2)
print('3' * 100)
# print(data.value_counts('BloodType'))
# ce_3=ce.BinaryEncoder(cols=['BloodType']).fit_transform(data)
# print('2'*100)
# print('\nBinaryEncoder Return the transformed dataset:\n', ce_3)
# print(data.value_counts('Education'))
# ce_5 = ce.HashingEncoder(cols=['Education']).fit_transform(data)
# print('\nReturn the transformed dataset:\n', ce_5)
# ce_6_1=ce.BaseNEncoder(cols=['BloodType'],base=3).fit_transform(data)
# print('\nBaseNEncoder Return the transformed dataset 1(base=3):\n', ce_6_1)

# ce_6_2 = ce.BaseNEncoder(cols=['BloodType'], base=4).fit_transform(data)
# print('6'*100)
# print('\nBaseNEncoder Return the transformed dataset 1(base=3):\n', ce_6_2)

# ce_7 = ce.SumEncoder(cols=['Education']).fit_transform(data)
# print('\nSumEncoder Return the transformed dataset:\n', ce_7)
# lr = ols('Income ~ Education_0 + Education_1 + Education_2',data=ce_7).fit()
# print(lr.summary())

# from statsmodels.formula.api import ols
# lr = ols('Income ~ Education_0 + Education_1 + Education_2', data=ce_7).fit()
# print(lr.summary())

# ce_8 = ce.BackwardDifferenceEncoder(cols=['Education']).fit_transform(data)
# print('\nBackwardDifferenceEncoder Return the transformed dataset:\n', ce_8)
# lr = ols('Income ~ Education_0 + Education_1 + Education_2',data=ce_8).fit()
# print(lr.summary())

# ce_9 = ce.HelmertEncoder(cols=['Education']).fit_transform(data)
# print('\nHelmertEncoder Return the transformed dataset:\n', ce_9)
# lr = ols('Income ~ Education_0 + Education_1 + Education_2', data=ce_9).fit()
# print(lr.summary())
# ce_12_1 = ce.MEstimateEncoder(cols=['Education'], random_state=10, m=0).fit_transform(data_new[features], data_new['Income'])
# print('12'*100)
# print('\nMEstimateEncoder Return the transformed dataset(m=0):\n', ce_12_1)

Income_grand_mean=data['Income'].mean()
print(Income_grand_mean)
data['Income_grand_mean'] = [Income_grand_mean]*len(data)
Income_group=data.groupby('Education')['Income'].mean().rename('Income_level_mean').reset_index()

data_new=pd.merge(data,Income_group)
