# -*- coding: utf-8 -*-
# @Time    : 2024/6/10 下午4:28
# @Author  : nanji
# @Site    : https://www.kaggle.com/code/abdmental01/lgbm-optimization-optuna
# @File    : solve02.py
# @Software: PyCharm 
# @Comment :

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math
import numpy as np
from colorama import Fore,Style,init
import optuna

import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score , train_test_split
