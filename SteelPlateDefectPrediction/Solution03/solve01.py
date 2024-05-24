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
		#  Changing original data column order to match the competition column structure:-
		self.original = self.original.reindex(self.train.columns, axis=1)

	def _AddSourceCol(self):
		self.train['Source'] = 'Competition'
		self.test['Source'] = 'Competition'
		self.original['Source'] = 'Original'
		self.strt_ftre = self.test.columns
		return self

	def _CollateInfoDesc(self):
		if self.dtl_preproc_req == 'Y':
			PrintColor(f"\n{'-' * 20} Information and description {'-' * 20}\n", color=Fore.MAGENTA);
			# Creating dataset information and description:
			for lbl, df in {'Train': self.train, 'Test': self.test, 'Original': self.original}.items():
				PrintColor(f"\n{lbl} description\n");
				# display(df.describe(percentiles=[0.05, 0.25, 0.50, 0.75, 0.9, 0.95, 0.99]). \
				# 		transpose(). \
				# 		drop(columns=['count'], errors='ignore'). \
				# 		drop([self.targets], axis=0, errors='ignore'). \
				# 		style.format(formatter='{:,.2f}'). \
				# 		background_gradient(cmap='Blues'))
				PrintColor(f"\n{lbl} information\n")
				# display(df.info());
				collect();
		return self

	def _CollateUnqNull(self):
		if self.dtl_preproc_req == 'Y':
			# Dislaying the unique values across train-test-original:-
			PrintColor(f"\nUnique and null values\n");
			_ = pd.concat([
				self.train[self.strt_ftre].nunique(),
				self.test[self.strt_ftre].nunique(),
				self.original[self.strt_ftre].nunique(),
				self.train[self.strt_ftre].isna().sum(axis=0),
				self.test[self.strt_ftre].isna().sum(axis=0),
				self.original[self.strt_ftre].isna().sum(axis=0)
			], axis=1)
			_, columns = ['Train_Nunq', 'Test_Nunq', 'Original_Nunq',
						  'Train_Nulls', 'Test_Nulls', 'Original_Nulls']
		# display(_.T.style.background_gradient(cmap='Blues', axis=1). \
		# 		format(formatter='{:,.0f}') );
		return self

	def _ConjoinTrainOrig(self):
		if self.conjoin_orig_data == 'Y':
			PrintColor(f"\n\nTrain shape before conjoining with original = {self.train.shape}");
			train = pd.cocnat([self.train, self.original], axis=0, ignore_index=True)
			train = train.drop_duplicates()
			PrintColor(f'Train shape after de-deupling ={train.shape}')
			train.index = range(len(train))
			train.index.name = 'id'
		else:
			PrintColor(f'\n We are using the competition training data only')
			train = self.train

	def DoPreprocessing(self):
		self._AddSourceCol()
		self._CollateInfoDesc()
		self._CollateUnqNull()
		self.train = self._ConjoinTrainOrig()
		self.train.index = range(len(self.train))
		_ = pp.train.drop(columns=CFG.targets + ['Source']).unique()
		self.cat_cols = _.loc[_ <= 10].index.to_list()
		self.cont_cols = [c for c in _.index if c not in self.cat_cols + ['Source']]
		return self


collect();
print();
pp = Preprocessor();
pp.DoPreprocessing();

print();
collect()


class FeaturePlotter(CFG, Preprocessor):
	'''
	This class develops plots for the targets,continuous and category features
	'''

	def __init__(self):
		super().__init__()
		clear_output()

	def MakeTgtPlot(self):
		'''
		This method returns the target plots
		:return:
		'''
		if self.ftre_plots_req == 'Y':
			for target in self.targets:
				fig, axes = plt.subplots(1, 2, figsize=(10, 3), \
										 sharey=True, \
										 gridspec_kw={'wspace': 0.35})
				for i, df in tqdm(enumerate([self.train, self.original]), f'Target plt - {target} --->'):
					ax = axes[i]
					a = df[target].value_counts(normalize=True)
					a.sort_index().plot.bar(color='tab:blue', ax=ax)
					df_name = 'Train' if i == 0 else 'Original'
					_ = ax.set_title(f"\n{df_name} data- {target}\n", **CFG.title_specs);
					ax.set_yticks(np.arange(0, 1.01, 0.08), labels=np.around(np.arange(0, 1.01, 0.08), 2), fontsize=7.0)
				plt.tight_layout();
				plt.show();

	def MakeCatFtrePlots(self, cat_cols):
		'''
		This method returns the category feature plots'
		:param cat_cols:
		:return:
		'''
		if cat_cols != [] and self.ftre_plots_req == 'Y':
			fig, axes = plt.subplots(len(cat_cols), 3, figsize=(20, len(cat_cols) * 4.5))
			for i, col in enumerate(cat_cols):
				ax = axes[i, 0]
				a = self.train[col].value_counts(normalize=True)
				a.sort_index().plot.barh(ax=ax, color='#007399')
				ax.set_title(f'{col}_Train ', **self.title_specs)
				ax.set_xticks(np.arange(0.0, 0.9, 0.05), \
							  labels=np.round(np.arange(0.0, 0.9, 0.05), 2), \
							  rotation=90)
				ax.set(xlabel='', ylabe='')
				del a
				ax = axes[i, 1];
				a = self.test[col].value_counts(normalize=True);
				a.sort_index().plot.barh(ax=ax, color='#0088cc');
				ax.set_title(f"{col}_Test", **self.title_specs);
				ax.set_xticks(np.arange(0.0, 0.9, 0.05),
							  labels=np.round(np.arange(0.0, 0.9, 0.05), 2),
							  rotation=90
							  );
				ax.set(xlabel='', ylabel='');
				del a;

				ax = axes[i, 2];
				a = self.original[col].value_counts(normalize=True);
				a.sort_index().plot.barh(ax=ax, color='#0047b3');
				ax.set_title(f"{col}_Original", **self.title_specs);
				ax.set_xticks(np.arange(0.0, 0.9, 0.05),
							  labels=np.round(np.arange(0.0, 0.9, 0.05), 2),
							  rotation=90
							  );
				ax.set(xlabel='', ylabel='');
				del a;
			plt.suptitle(f"Category column plots", **self.title_specs, y=0.94);
			plt.tight_layout();
			plt.show();

	def MakeContCol(self, cont_cols):
		'''
		    This method returns the continuous feature plots
		:param cont_cols:
		:return:
		'''
		if self.ftre_plots_req == 'Y':
			df = pd.concat(
				[
					self.train[cont_cols].assign(Source='Train'),
					self.test[cont_cols].assign(Source='Test'),
					self.original[cont_cols].assign(Source='Original'),
				],
				axis=0, ignore_index=True
			)
			fig, axes = plt.subplots(len(cont_cols), 4, figsize=(16, len(cont_cols) * 4.2),
									 gridspec_kw={'hspace': 0.35,
												  'wspace': 0.3,
												  'width_ratios': [0.80, 0.20, 0.20, 0.20]
												  }
									 )
			for i, col in enumerate(cont_cols):
				ax = axes[i, 0];
				sns.kdeplot(data=df[[col, 'Source']], x=col, hue='Source',
							palette=['#0039e6', '#ff5500', '#00b300'],
							ax=ax, linewidth=2.1
							);
				ax.set_title(f"\n{col}", **self.title_specs);
				ax.grid(**CFG.grid_specs);
				ax.set(xlabel='', ylabel='');

				ax = axes[i, 1];
				sns.boxplot(data=df.loc[df.Source == 'Train', [col]], y=col, width=0.25,
							color='#33ccff', saturation=0.90, linewidth=0.90,
							fliersize=2.25,
							ax=ax);
				ax.set(xlabel='', ylabel='');
				ax.set_title(f"Train", **self.title_specs);

				ax = axes[i, 2];
				sns.boxplot(data=df.loc[df.Source == 'Test', [col]], y=col, width=0.25, fliersize=2.25,
							color='#80ffff', saturation=0.6, linewidth=0.90,
							ax=ax);
				ax.set(xlabel='', ylabel='');
				ax.set_title(f"Test", **self.title_specs);
				ax = axes[i, 3];
				sns.boxplot(data=df.loc[df.Source == 'Original', [col]], y=col, width=0.25, fliersize=2.25,
							color='#99ddff', saturation=0.6, linewidth=0.90,
							ax=ax);
				ax.set(xlabel='', ylabel='');
				ax.set_title(f"Original", **self.title_specs);

			plt.suptitle(f"\nDistribution analysis- continuous columns\n", **CFG.title_specs,
						 y=0.89, x=0.50
						 );
			plt.tight_layout();
			plt.show();

	def CalcSkew(self, cont_cols):
		"This method calculates the skewness across columns";
		if self.ftre_plots_req == "Y":
			skew_df = pd.DataFrame(index=cont_cols);
			for col, df in {"Train": self.train[cont_cols],
							"Test": self.test[cont_cols],
							"Original": self.original[cont_cols]
							}.items():
				skew_df = \
					pd.concat([skew_df,
							   df.drop(columns=self.targets + ["Source", "id"], errors="ignore").skew()],
							  axis=1).rename({0: col}, axis=1);

			PrintColor(f"\nSkewness across independent features\n");


# display(skew_df.transpose().style.format(precision=2).background_gradient("PuBuGn"));


print();
collect();
plotter = FeaturePlotter();
plotter.MakeCatFtrePlots(cat_cols=pp.cat_cols)

print(f"\n\n\n");
plotter.MakeContColPlots(cont_cols=pp.cont_cols);

print(f"\n\n\n");
plotter.MakeTgtPlot();

print(f"\n\n\n");
plotter.CalcSkew(cont_cols=pp.cont_cols);

print();
collect();


class Xformer(TransformerMixin, BaseEstimator):
	"""
	This class adds secondary features to the existing data using simple interactions
	"""

	def __init__(self):
		self.sec_ftre_req = CFG.sec_ftre_req

	def fit(self, X, y=None, **params):
		return self

	@staticmethod
	def _reduce_mem(df: pd.DataFrame):
		"This method reduces memory for numeric columns in the dataframe";
		numerics = ['int16', 'int32', \
					'int64', 'float16', \
					'float32', 'float64', \
					"uint16", "uint32", "uint64"];
		start_mem = df.memory_usage().sum() / 1024 ** 2
		for col in df.columns:
			col_type = df[col].dtypes
			if col_type in numerics:
				c_min = df[col].min()
				c_max = df[col].max()
				if 'int' in str(col_type):
					if c_min >= np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
						df[col] = df[col].astype(np.int8)
					elif c_min >= np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
						df[col] = df[col].astype(np.int16)
					elif c_min >= np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
						df[col] = df[col].astype(np.int32)
					elif c_min >= np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
						df[col] = df[col].astype(np.int64)
				else:
					if c_min >= np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
						df[col] = df[col].astype(np.float16)
					if c_min >= np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
						df[col] = df[col].astype(np.float32)
					else:
						df[col] = df[col].astype(np.float64)
		end_mem = df.memory_usage().sum() / 1024 ** 2
		PrintColor(f'Start - end memory: - {start_mem:5.2f} - {end_mem:5.2f} Mb')
		return df

	def transform(self, X, y=None, **params):
		'''
        This method adds secondary features to the existing data
        Source:- https://www.kaggle.com/code/lucamassaron/steel-plate-eda-xgboost-is-all-you-need
		:param X:
		:param y:
		:param params:
		:return:
		'''
		df = X.copy()
		if self.sec_ftre_req == 'Y':
			df['XRange'] = df['X_Maximum'] - df['X_Minimum']
			df['YRange'] = df['Y_Maximum'] - df['Y_Minimum']
			df['Area_Perimeter_Ratio'] = df['Pixels_Areas'] / (df['X_Perimeter'] + df['Y_Perimeter'])
			df['Aspect_Ratio'] = np.where(df['YRange'] == 0, 0, df['XRange'] / df['YRange']);
		self.op_cols = df.columns
		df = self._reduce_mem(df)
		return df

	def get_feature_names_in(self, X, y=None, **params):
		return self.ip_cols

	def get_feature_names_out(self, X, y=None, **params):
		return self.op_cols


collect()
print()
