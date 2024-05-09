# -*- coding: utf-8 -*-
# @Time    : 2024/5/9 下午9:49
# @Author  : nanji
# @Site    :https://blog.csdn.net/dss_dssssd/article/details/82824979
# @File    : testTransformerMixin01.py
# @Software: PyCharm 
# @Comment :
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np

room_ix, bedroom_ix, population_ix, household_ix = 3, 4, 5, 6
print('0' * 100)
print(household_ix[:5])


class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
	def __init__(self, add_bedrooms_per_root=True):
		self.add_bedrooms_per_room = add_bedrooms_per_root
		super().__init__()

	def fit(self, X, y=None):
		return self  #

	def transform(self, X, y=None):
		rooms_per_household = X[:, room_ix] / X[:, household_ix]
		population_per_household = X[:, population_ix] / X[:, household_ix]
		if self.add_bedrooms_per_room:
			bedroom_per_room = X[:, bedroom_ix] / X[:, room_ix]
			return np.c_[X, rooms_per_household, population_per_household, bedroom_per_room]
		else:
			return np.c_[X, rooms_per_household, population_per_household]

attr_adder=CombinedAttributesAdder(add_bedrooms_per_root=False)
housing_extra_attribs=attr_adder.transform(housing.values)

