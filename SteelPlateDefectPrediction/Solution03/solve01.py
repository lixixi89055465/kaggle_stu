# -*- coding: utf-8 -*-
# @Time : 2024/5/18 20:25
# @Author : nanji
# @Site : https://www.kaggle.com/code/ravi20076/playgrounds4e03-eda-binaryclassifier
# @File : solve01.py
# @Software: PyCharm 
# @Comment :
from gc import collect;
from warnings import filterwarnings;

filterwarnings('ignore')
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
from category_encoders import OrdinalEncoder, OneHotEncoder
from sklearn.preprocessing import (RobustScaler, MinMaxScaler, \
								   StandardScaler, \
								   FunctionTransformer as FT, \
								   PowerTransformer)
from sklearn.impute import SimpleImputer as SI
from sklearn.model_selection import (RepeatedStratifiedKFold as RSKF, \
									 StratifiedKFold as SKF, \
									 StratifiedGroupKFold as SGKF, \
									 KFold, \
									 RepeatedKFold as RKF, \
									 cross_val_score, \
									 cross_val_predict)
from sklearn.inspection import permutation_importance
from sklearn.feature_selection import mutual_info_classif, RFE
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer

# ML Model training : ~
from sklearn.metrics import accuracy_score, roc_auc_score, make_scorer
from xgboost import DMatrix, XGBClassifier as XGBC
from lightgbm import log_evaluation, early_stopping, LGBMClassifier as LGBMC
from catboost import CatBoostClassifier as CBC, Pool
from sklearn.ensemble import HistGradientBoostingClassifier as HGBC, \
	RandomForestClassifier as RFC

# Ensemble and tuning
import optuna
from optuna import Trial, trial, create_study
from optuna.pruners import HyperbandPruner
from optuna.samplers import TPESampler, CmaEsSampler

optuna.logging.set_verbosity = optuna.logging.ERROR
clear_output()
print()
collect()

# Setting rc parameters in seaborn for plots and graphs-
# Reference - https://matplotlib.org/stable/tutorials/introductory/customizing.html:-
# To alter this, refer to matplotlib.rcParams.keys()

sns.set({"axes.facecolor": "#ffffff",
		 "figure.facecolor": "#ffffff",
		 "axes.edgecolor": "#000000",
		 "grid.color": "#ffffff",
		 "font.family": ['Cambria'],
		 "axes.labelcolor": "#000000",
		 "xtick.color": "#000000",
		 "ytick.color": "#000000",
		 "grid.linewidth": 0.75,
		 "grid.linestyle": "--",
		 "axes.titlecolor": '#0099e6',
		 'axes.titlesize': 8.5,
		 'axes.labelweight': "bold",
		 'legend.fontsize': 7.0,
		 'legend.title_fontsize': 7.0,
		 'font.size': 7.5,
		 'xtick.labelsize': 7.5,
		 'ytick.labelsize': 7.5,
		 });


# Color printing

# Color printing
def PrintColor(text: str, color=Fore.BLUE, style=Style.BRIGHT):
	"Prints color outputs using colorama using a text F-string";
	print(style + color + text + Style.RESET_ALL);


# Making sklearn pipeline outputs as dataframe:-
from sklearn import set_config

# set_config(transform_output ="pandas");


pd.set_option('display.max_columns', 50);
pd.set_option('display.max_rows', 50);

print('0' * 100)
print()
collect()


class CFG:
	"""
	Configuration class for parameters and CV strategy for tuning and training
	Some parameters may be unused here as this is a general configuration class
	""";
	# Data preparation
	version_nb = 7
	test_req = 'Y'
	test_sample_frac = 0.025
	gpu_switch = 'OFF'
	state = 42
	targets = ['Pastry', 'Z_Scratch', \
			   'K_Scatch', 'Stains', 'Dirtiness', \
			   'Bumps', 'Other_Faults']
	episode = 3
	season = 4
	path = f"../input/playground-series-s{season}e{episode}";
	orig_path = f"../input/playgrounds4e03ancillary/PlaygroundS4E3Original.csv";
	dtl_preproc_req = 'Y'
	ftre_plots_req = 'Y'
	ftre_imp_req = 'Y'

	# Data transforms and scaling
	conjoin_orig_data = 'Y'
	drop_nulls = 'N'
	sec_ftre_req = 'Y'
	scale_req = 'N'
	# Model Training:-
	pstprcs_oof = 'N'
	pstprcs_train = 'N'
	pstprcs_test = 'N'
	ML = 'Y'

	pseudo_lbl_req = 'N'
	pseudolbl_up = 0.975
	pseudolbl_low = 0.00

	n_splits = 3 if test_req == 'Y' else 10
	n_repeats = 1
	nbrnd_erly_stp = 75
	mdlcv_mthd = 'RSKF'

	# Ensemble : -
	ensemble_req = 'Y'
	hill_climb_req = 'N'
	optuna_req = 'Y'
	LAD_req = 'N'
	enscv_mthd = 'RSKF'
	metric_obj = 'maximize'
	ntrials = 10 if test_req == 'Y' else 100
	# Global variables for plotting:-
	grid_specs = {'visible': True, 'which': 'both', 'linestyle': '--',
				  'color': 'lightgrey', 'linewidth': 0.75};
	title_specs = {'fontsize': 9, 'fontweight': 'bold', 'color': '#992600'};


print()
PrintColor(f'--> Configuration done! \n')
collect()


class Preprocessor():
	"""
	This class aims to do the below-
	1. Read the datasets
	2. In this case, process the original data
	3. Check information and description
	4. Check unique values and nulls
	5. Collate starting features
	6. Conjoin train-original data if requested based on Adversarial CV results
	""";

	def __init__(self):
		self.train = pd.read_csv(os.path.join(CFG.path, 'train.csv'), index_col='id')
		self.test = pd.read_csv(os.path.oin(CFG.path, 'test.csv'), index_col='id')
		self.targets = CFG.targets
		self.original = pd.read_csv(CFG.orig_path, index_col='id')
		self.conjoin_orig_data = CFG.conjoin_orig_data
		self.dtl_preproc_req = CFG.dtl_preproc_req
		self.test_req = CFG.test_req
		self.sub_f1 = pd.read_csv(os.path.join(CFG.path, 'sample_submission.csv'))
		PrintColor(f"Data shapes - train-test-original | {self.train.shape} {self.test.shape} {self.original.shape}");
		for tbl in [self.train, self.original, self.test]:
			tbl.columns = tbl.columns.str.replace(r"\(|\)|\s+", "", regex=True);

		# PrintColor(f"\nTrain set head", color=Fore.CYAN);
		# display(self.train.head(5).style.format(precision=3));
		# PrintColor(f"\nTest set head", color=Fore.CYAN);
		# display(self.test.head(5).style.format(precision=3));
		# PrintColor(f"\nOriginal set head", color=Fore.CYAN);
		# display(self.original.head(5).style.format(precision=3));
		# Resetting original data index:-
		self.original.index = range(len(self.original));
		self.original.index += max(self.test.index) + 1;
		self.original.index.name = 'id';
