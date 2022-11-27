'''
'''
import pandas as pd
import numpy as np
from scipy import stats
from mlxtend.preprocessing import minmax_scaling

import seaborn as sns
import matplotlib.pyplot as plt

kickstarters_2017 = pd.read_csv("./data/kickstarter-projects/ks-projects-201801.csv")
# set seed for reproducibility
np.random.seed(0)

original_data = pd.DataFrame(kickstarters_2017.usd_goal_real)
scaled_data = minmax_scaling(original_data, columns=['usd_goal_real'])
# print('Original data \n Preview :\n', original_data.head())
# print(
#     'Minimum value:', float(original_data.min()),
#     '\nMaximum value:', float(original_data.max()) )
# print('_'*30)
# print('\n Scaled data \n Preview :\n',scaled_data.head())
# print('Minimum value:',float(scaled_data.min()),
#       '\nMaximum value:', float(scaled_data.max()))

original_goal_data = pd.DataFrame(kickstarters_2017.goal)
scaled_goal_data = minmax_scaling(original_goal_data, columns=['goal'])

# print(kickstarters_2017.columns)
index_of_positive_pledges=kickstarters_2017.usd_pledged_real>0
positive_pledges=kickstarters_2017.usd_pledged_real.loc[index_of_positive_pledges]
normalized_pledges=pd.Series(stats.boxcox(positive_pledges)[0],name='usd_pledged_real',index=positive_pledges.index)

##
index_of_positive_pledges=kickstarters_2017.pledged>0
positive_pledges=kickstarters_2017.pledged.loc[index_of_positive_pledges]
normalized_pledges=pd.Series(stats.boxcox(positive_pledges)[0],name='pledged',index=positive_pledges.index)

