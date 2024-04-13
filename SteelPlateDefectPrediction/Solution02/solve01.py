# -*- coding: utf-8 -*-
# @Time : 2024/4/5 16:04
# @Author : nanji
# @Site : https://www.kaggle.com/code/arunklenin/ps4e3-steel-plate-fault-prediction-multilabel
# @File : solve01.py
# @Software: PyCharm 
# @Comment : PS4E3 | Steel Plate Fault Prediction |Multilabel
import sklearn
import numpy as np
import os
import datetime
import pandas as pd
import matplotlib.pyplot as plt
import missingno as msno
from sklearn.datasets import load_boston
import pandas as pd
import numpy as np
import warnings
from prettytable import PrettyTable
import seaborn as sns
sns.set(style='darkgrid', font_scale=1.4)
from tqdm import tqdm
from tqdm.notebook import tqdm as tqdm_notebook
tqdm_notebook.get_lock().locks = []
import concurrent.futures
from copy import deepcopy

from functools import partial
from itertools import combinations


print('0'*100)














warnings.filterwarnings('ignore')

data, target = load_boston()['data'], load_boston()['target']
df = pd.concat(
	[pd.DataFrame(data, columns=[str('features' + str(i)) for i in range(13)]),
	 pd.DataFrame(target, columns=['target'])], axis=1)

df1 = df.where(df != 0, np.nan)
print(df1.isnull().sum())
print('0'*100)
msno.matrix(df1, labels=True)