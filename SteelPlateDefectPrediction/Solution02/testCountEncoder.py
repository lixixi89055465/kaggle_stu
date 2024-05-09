# -*- coding: utf-8 -*-
# @Time    : 2024/5/9 下午10:44
# @Author  : nanji
# @Site    : https://www.cnblogs.com/dangui/p/15836197.html#4%E8%AE%A1%E6%95%B0%E7%BC%96%E7%A0%81countencoder
# @File    : testCountEncoder.py
# @Software: PyCharm 
# @Comment :


import category_encoders as ce
import pandas as pd

data = pd.DataFrame({'ID': [1, 2, 3, 4, 5, 6, 7, 8], \
					 'Sex': ['F', 'M', 'M', 'F', 'M', None, 'F', 'M'], \
					 'BloodType': ['A', 'AB', 'O', 'B', None, 'O', 'AB', 'B'], \
					 'Grade': ['High', None, 'Medium', 'Low', 'Low', 'Medium', 'Low', 'High'], \
					 # 'Height':[156, None, 167, 175, 164, 180], 'Weight':[50, None, 65, 67, 48, 76],
					 'Education': ['PhD', 'HighSchool', 'Bachelor', 'Master', 'HighSchool', 'Master', 'PhD',
								   'Bachelor'], \
					 'Income': [28300, 4500, 7500, 12500, 4200, 15000, 25000, 7200]})

print('\nOriginal Dataset:\n', data)

ce_1 = ce.OrdinalEncoder(cols=['Grade', 'Education'],
mapping = [
	{
		'col': 'Grade', \
		'mapping': {None: 0, 'Low': 1, 'Medium': 2, 'High': 3} \
		}, \
	{
		'col': 'Education', \
		'mapping': {None: 0, 'HighSchool': 1, 'Bachelor': 2, 'Master': 3, 'PhD': 4} \
		}
]).fit_transform(data)
print('\nOrdinalEncoder Return the transformed dataset:\n', ce_1)
