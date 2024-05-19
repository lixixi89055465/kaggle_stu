# -*- coding: utf-8 -*-
# @Time : 2024/5/18 20:25
# @Author : nanji
# @Site : https://www.kaggle.com/code/ravi20076/playgrounds4e03-eda-binaryclassifier
# @File : solve01.py
# @Software: PyCharm 
# @Comment :
from gc import collect;
from warnings import filterwarnings;

filterwarnings('ignore');
from IPython.display import display_html, clear_output;

clear_output();
import xgboost as xgb, lightgbm as lgb, catboost as cb, sklearn as sk, pandas as pd;

print(f"---> XGBoost = {xgb.__version__} | LightGBM = {lgb.__version__} | Catboost = {cb.__version__}");
print(f"---> Sklearn = {sk.__version__}| Pandas = {pd.__version__}\n\n");
collect();
collect();
from copy import deepcopy
import pandas as pd
import numpy as np
import re
from scipy.stats import mode, kstest, normaltest, shapiro, anderson, jarque_bera
from collections import Counter
from itertools import product
from colorama import Fore, Style, init
from warnings import filterwarnings

filterwarnings('ignore')
import joblib
import os

from tqdm.notebook import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap as LCM
from pprint import pprint
from functools import partial

print(print())
print(collect())
print(clear_output())
