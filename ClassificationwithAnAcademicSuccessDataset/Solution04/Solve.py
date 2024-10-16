# -*- coding: utf-8 -*-
# @Time : 2024/10/16 22:25
# @Author : nanji
# @Site : https://www.kaggle.com/code/endofnight17j03/academicsuccess-voting-xgboost
# @File : Solve.py
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
