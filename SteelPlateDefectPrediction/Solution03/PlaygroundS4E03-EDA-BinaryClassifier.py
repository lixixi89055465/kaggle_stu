# -*- coding: utf-8 -*-
# @Time : 2024/5/18 10:14
# @Author : nanji
# @Site :https://www.kaggle.com/code/ravi20076/playgrounds4e03-eda-binaryclassifier
# @File : PlaygroundS4E03-EDA-BinaryClassifier.py
# @Software: PyCharm 
# @Comment :

# Installing select libraries:-
from gc import collect;
from warnings import filterwarnings;

filterwarnings('ignore');
from IPython.display import display_html, clear_output;

clear_output();
import xgboost as xgb, lightgbm as lgb, catboost as cb, sklearn as sk, pandas as pd;

print(f"---> XGBoost = {xgb.__version__} | LightGBM = {lgb.__version__} | Catboost = {cb.__version__}");
print(f"---> Sklearn = {sk.__version__}| Pandas = {pd.__version__}\n\n");
collect();
from copy import deepcopy
import pandas as pd
import numpy as np
import re
from scipy.stats import mode, kstest, normaltest, shapiro, anderson, jarque_bera

