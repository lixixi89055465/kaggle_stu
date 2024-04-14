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
from sklearn.feature_selection import f_classif

from sklearn.preprocessing import LabelEncoder, StandardScaler, \
	MinMaxScaler, PowerTransformer, FunctionTransformer
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from itertools import combinations
from sklearn.impute import SimpleImputer
import xgboost as xg
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, \
	mean_squared_log_error, \
	roc_auc_score, \
	accuracy_score, \
	f1_score,\
	precision_recall_curve,\
	log_loss
from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer
from gap_statistic.optimalK import OptimalK
from scipy import stats
import statsmodels.api as sm

from scipy.stats import ttest_ind
from scipy.stats import boxcox

import math
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.base import BaseEstimator, TransformerMixin
import optuna
import cmaes
# pip install cmaes
import xgboost as xgb
import lightgbm as lgb
from category_encoders import OneHotEncoder, OrdinalEncoder, CountEncoder, CatBoostEncoder
from imblearn.under_sampling import RandomUnderSampler



