# -*- coding: utf-8 -*-
# @Time : 2024/10/16 22:25
# @Author : nanji
# @Site : https://www.kaggle.com/code/endofnight17j03/academicsuccess-voting-xgboost
# @File : Solve01.py
# @Software: PyCharm 
# @Comment :
import statsmodels.api as sm
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from sklearn.ensemble import (GradientBoostingClassifier,  #
                              RandomForestClassifier,  #
                              AdaBoostClassifier,  #
                              VotingClassifier,  #
                              HistGradientBoostingClassifier)
from sklearn.metrics import accuracy_score, r2_score, recall_score, roc_auc_score, \
    confusion_matrix, classification_report
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import lightgbm as lgb
import warnings
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.exceptions import DataConversionWarning
from catboost import CatBoostClassifier, Pool
from catboost.utils import eval_metric

warnings.filterwarnings('ignore', category=DataConversionWarning)
warnings.filterwarnings('ignore')
from IPython.display import display, HTML
from sklearn.metrics import auc
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import cross_val_score
from catboost import CatBoostClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.tree import DecisionTreeClassifier
from scipy.stats import ttest_ind, chi2_contingency
from sklearn.naive_bayes import GaussianNB
from sklearn.utils import class_weight
import pandas as pd
from sklearn.preprocessing import PowerTransformer, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from scipy.stats import skew
