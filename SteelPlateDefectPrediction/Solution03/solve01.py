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
		self.test = pd.read_csv(os.path.join(CFG.path, 'test.csv'), index_col='id')
		self.targets = CFG.targets
		self.original = pd.read_csv(CFG.orig_path, index_col='id')
		self.conjoin_orig_data = CFG.conjoin_orig_data
		self.dtl_preproc_req = CFG.dtl_preproc_req
		self.test_req = CFG.test_req
		self.sub_f1 = pd.read_csv(os.path.join(CFG.path, 'sample_submission.csv'))
		PrintColor(f"Data shapes - train-test-original | {self.train.shape} {self.test.shape} {self.original.shape}");
		# 去除列columns中的特殊字符
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
			_.columns = ['Train_Nunq', 'Test_Nunq', 'Original_Nunq',
						 'Train_Nulls', 'Test_Nulls', 'Original_Nulls']
		# display(_.T.style.background_gradient(cmap='Blues', axis=1). \
		# 		format(formatter='{:,.0f}') );
		return self

	def _ConjoinTrainOrig(self):
		if self.conjoin_orig_data == 'Y':
			PrintColor(f"\n\nTrain shape before conjoining with original = {self.train.shape}");
			train = pd.concat([self.train, self.original], axis=0, ignore_index=True)
			PrintColor(f'Train shape after de-deupling ={train.shape}')

			train = train.drop_duplicates()
			PrintColor(f'Train shape after de-deupling ={train.shape}')
			train.index = range(len(train))
			train.index.name = 'id'
		else:
			PrintColor(f'\n We are using the competition training data only')
			train = self.train
		return train

	def DoPreprocessing(self):
		self._AddSourceCol()
		self._CollateInfoDesc()
		self._CollateUnqNull()
		self.train = self._ConjoinTrainOrig()
		self.train.index = range(len(self.train))
		_ = pp.train.drop(columns=CFG.targets + ['Source']).nunique()
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
				ax.set(xlabel='', ylabel='')
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

	def MakeContColPlots(self, cont_cols):
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

PrintColor(f"\n{'=' * 20} Data transformation {'=' * 20} \n");
ytrain = pp.train[CFG.targets]

xform = Pipeline(steps=[('xfrm', Xformer())])
Xtrain = xform.fit_transform(pp.train.drop(columns=CFG.targets))
Xtest = xform.transform(pp.test.copy(deep=True))

PrintColor(f'\n --->Train data \n ')
# display(Xtrain.head(5).style.format(precision = 2));
PrintColor(f'\n---> Test data\n')
# display(.head(5).style.format(precision = 2));
# display(Xtest.head(5).style.format(precision = 2));

# Checking the results:-
with np.printoptions(linewidth=160):
	PrintColor(f"\n---> Train data columns after data pipeline\n");
	pprint(np.array(Xtrain.columns))
	PrintColor(f"\n---> Test data columns after data pipeline\n");
	pprint(np.array(Xtest.columns))
	PrintColor(f"\n---> Train-test shape after pipeline = {Xtrain.shape} {Xtest.shape}");

print()
collect()


class OptunaEnsembler:
	'''
	  This is the Optuna ensemble class-
    Source- https://www.kaggle.com/code/arunklenin/ps3e26-cirrhosis-survial-prediction-multiclass
	'''

	def __init__(self):
		self.study = None
		self.weights = None
		self.random_state = CFG.state
		self.n_trials = CFG.ntrials
		self.direction = CFG.metric_obj

	def ScoreMetric(self, ytrue, ypred):
		'''
		This is the metric function for the competition
		:param ytrue:
		:param ypred:
		:return:
		'''
		return roc_auc_score(ytrue, ypred)

	def _object(self, trial, y_true, y_preds):
		''''
        This method defines the objective function for the ensemble
		'''
		if isinstance(y_preds, pd.DataFrame) or isinstance(y_preds, np.ndarray):
			weights = [trial.suggest_float(f'weight{n}', 0, 1)
					   for n in range(y_preds.shape[-1])]
			axis = 1
		elif isinstance(y_preds, list):
			weights = [trial.suggest_float(f'weight{n}', 0, 1) \
					   for n in range(len(y_preds))]
			axis = 0
		weighted_pred = np.average(np.array(y_preds), axis=axis, weights=weights)
		score = self.ScoreMetric(y_true, weighted_pred)
		return score

	def fit(self, y_true, y_preds):
		'''
		This method fits the Optuna objective on the fold level data
		:param y_true:
		:param y_preds:
		:return:
		'''
		optuna.logging.set_verbosity = optuna.logging.ERROR
		self.study = optuna.create_study(
			sampler=TPESampler(self=self.random_state),
			pruner=HyperbandPruner(),
			study_name='Ensemble',
			direction=self.direction
		)
		obj = partial(self._objective, y_true=y_true, y_preds=y_preds)
		self.study.optimize(obj, n_trials=self.n_trials)
		if isinstance(y_preds, list):
			self.weights = [self.study.best_params[f'weight{n}'] \
							for n in range(len(y_preds))]
		else:
			self.weights = [self.study.best_params[f'weight{n}'] \
							for n in range(y_preds.shape[-1])]
		clear_output()

	def predict(self, y_preds):
		'''
		This method predicts using the fitted Optuna objective;
		:param y_preds:
		:return:
		'''
		assert self.weights is not None, 'OptunaWeights error, must be fitted before predict';
		if isinstance(y_preds, list):
			weighted_pred = np.average(np.array(y_preds), axis=0, weights=self.weights)
		else:
			weighted_pred = np.average(np.array(y_preds), axis=1, weights=self.weights)
		return weighted_pred

	def fit_predict(self, y_true, y_pres):
		self.fit(y_true, y_pres)
		return self.predict(y_pres)

	def weights(self):
		return self.weights


print()
collect()


class MdlDeveloper(CFG):
	'''
	This class implements the training pipeline elements-
    1. Initializes the Model predictions
    2. Trains and infers models
    3. Returns the OOF and model test set predictions
	'''

	def __init__(self, Xtrain, ytrain, ygrp, Xtest, \
				 sel_cols, cat_cols, enc_cols, **kwargs):
		'''
		In this method,we initialize the below -
		1.Train-test data,selected columns
		2.Metric,custom scorer,model and cv object
		4.Output tables for score and predictions
		:param Xtrain:
		:param ytrain:
		:param ygrp:
		:param Xtest:
		:param sel_cols:
		:param cat_cols:
		:param enc_cols:
		:param kwargs:
		'''
		self.Xtrain = Xtrain
		self.ytrain = ytrain
		self.y_grp = ygrp
		self.Xtest = Xtest
		self.sel_cols = sel_cols
		self.cat_cols = cat_cols
		self.enc_cols = enc_cols
		self._DefineModel()
		self.cv = self.all_cv[self.mdlcv_mthd]
		self.methods = list(self.Mdl_Master.keys())
		self.OOF_Preds = pd.DataFrame()
		self.Mdl_Preds = pd.DataFrame()
		self.Scores = pd.DataFrame( \
			columns=self.methods + ['Ensemble'], \
			index=range(self.n_splits * self.n_repeats))
		self.TrainScores = pd.DataFrame( \
			columns=self.methods, \
			index=range(self.n_splits * self.n_repeats) \
			)
		self.mdlscorer = make_scorer( \
			self.ScoreMetric, \
			greater_is_better=True, \
			needs_proba=True, \
			needs_threshold=False
		)
		PrintColor(f'\n ---> Selected model option- ')
		try:
			with np.printoptions(linewidth=150):
				pprint(np.array(self.methods), depth=1, width=100, indent=5)

		except:
			pprint(self.methods, depth=1, width=100, indent=5)

	def _DefineModel(self):
		'''
		This method initiliazes models for the analysis
        It also initializes the CV methods and class-weights that could be tuned going ahead.
		:return:
		'''
		# Commonly used CV strategies for later usage:
		self.all_cv = {
			'KF': KFold(n_splits=self.n_splits, shuffle=True, random_state=self.state),
			'RKF': RKF(n_splits=self.n_splits, n_repeats=self.n_repeats, random_state=self.state),
			'RSKF': RSKF(n_splits=self.n_splits, n_repeats=self.n_repeats, random_state=self.state),
			'SKF': SKF(n_splits=self.n_splits, shuffle=True, random_state=self.state),
			'SGKF': SGKF(n_splits=self.n_splits, shuffle=True, random_state=self.state)
		}
		self.Mdl_Master = {
			'XGB1C': XGBC(
				**{
					'tree_method': 'hist',
					'device': 'cuda' if self.gpu_switch == 'ON' else 'cpu',
					'objective': 'binary:logistic',
					'eval_metric': 'auc',
					'random_state': self.state,
					'colsample_bytree': 0.25,
					'learning_rate': 0.07,
					'max_depth': 8,
					'n_estimcator': 1100,
					'reg_alpha': 0.7,
					'reg_lambda': 0.7,
					'min_child_weight': 22,
					'early_stopping_rounds': self.nbrnd_erly_stp,
					'verbosity': 0,
					'enable_categorical': True
				}),
			'XGB2C': XGBC(**{
				'tree_method': 'hist',
				'device': 'cuda' if self.gpu_switch == 'ON' else 'cpu',
				'eval_metric': 'auc',
				'random_state': self.state,
				'colsample_bytree': 0.4,
				'learning_rate': 0.06,
				'max_depth': 9,
				'n_estimator': 2500,
				'reg_alpha': 0.12,
				'reg_lambda': 0.8,
				'min_child_weight': 15,
				'early_stopping_rounds': self.nbrnd_erly_stp,
				'verbosity': 0,
				'enable_categorical': True,
			}),
			'XGB3C': XGBC(**{
				'tree_method': 'hist',
				'device': 'cuda' if self.gpu_switch == 'ON' else 'cpu',
				'objective': 'binary:logistic',
				'eval_metric': 'auc',
				'random_state': self.state,
				'colsample_bytree': 0.5,
				'learning_rate': 0.055,
				'max_depth': 9,
				'n_estimator': 3000,
				'reg_alpha': .2,
				'reg_lambda': 0.6,
				'min_child_weight': 25,
				'early_stopping_rounds': CFG.nbrnd_erly_stp,
				'verbosity': 0,
				'enable_categorical': True,
			}),
			'XGB4C': XGBC(**{
				'tree_method': 'hist',
				'device': 'cuda' if self.gpu_switch == 'ON' else 'cpu',
				'eval_metric': 'auc',
				'random_state': self.state,
				'colsample_bytree': 0.8,
				'learning_rate': 0.082,
				'max_depth': 7,
				'n_estimator': 2000,
				'reg_alpha': 0.005,
				'reg_lambda': 0.95,
				'min_child_weight': 26,
				'early_stopping_round': self.nbrnd_erly_stp,
				'verbosity': 0,
				'enable_categorical': True
			}),
			'LGBM1C': LGBMC(**{
				'device': 'gpu' if self.gpu_switch == 'ON' else 'cpu',
				'objective': 'binary',
				'boosting_type': 'gbdt',
				'metric': 'auc',
				'random_state': self.state,
				'colsample_bytree': 0.56,
				'subsample': 0.35,
				'learning_rate': 0.05,
				'max_depth': 6,
				'n_estimators': 3000,
				'num_leaves': 140,
				'reg_alpha': 0.14,
				'reg_lambda': 0.85,
				'verbosity': -1,
				'categorical_feature': [f'name:{c}' for c in self.cat_cols]
			}),
			'LGBM2C': LGBMC(**{'device': "gpu" if self.gpu_switch == "ON" else "cpu",
							   'objective': 'binary',
							   'boosting_type': 'gbdt',
							   'data_sample_strategy': "goss",
							   'metric': "auc",
							   'random_state': self.state,
							   'colsample_bytree': 0.20,
							   'subsample': 0.25,
							   'learning_rate': 0.10,
							   'max_depth': 7,
							   'n_estimators': 3000,
							   'num_leaves': 120,
							   'reg_alpha': 0.15,
							   'reg_lambda': 0.90,
							   'verbosity': -1,
							   'categorical_feature': [f"name: {c}" for c in self.cat_cols],
							   }
							),

			'LGBM3C': LGBMC(**{'device': "gpu" if CFG.gpu_switch == "ON" else "cpu",
							   'objective': 'binary',
							   'boosting_type': 'gbdt',
							   'metric': "auc",
							   'random_state': self.state,
							   'colsample_bytree': 0.45,
							   'subsample': 0.45,
							   'learning_rate': 0.06,
							   'max_depth': 6,
							   'n_estimators': 3000,
							   'num_leaves': 125,
							   'reg_alpha': 0.05,
							   'reg_lambda': 0.95,
							   'verbosity': -1,
							   'categorical_feature': [f"name: {c}" for c in self.cat_cols],
							   }
							),

			'LGBM4C': LGBMC(**{'device': "gpu" if self.gpu_switch == "ON" else "cpu",
							   'objective': 'binary',
							   'boosting_type': 'gbdt',
							   'metric': "auc",
							   'random_state': self.state,
							   'colsample_bytree': 0.55,
							   'subsample': 0.55,
							   'learning_rate': 0.085,
							   'max_depth': 7,
							   'n_estimators': 3000,
							   'num_leaves': 105,
							   'reg_alpha': 0.08,
							   'reg_lambda': 0.995,
							   'verbosity': -1,
							   }
							),
			'CB1C': CBC(**{
				'task_type': 'GPU' if self.gpu_switch == 'ON' else 'CPU',
				'objective': 'Logloss',
				'eval_metric': 'AUC',
				'bagging_temperature': 0.1,
				'colsample_bylevel': 0.88,
				'iterations': 3000,
				'learning_rate': 0.065,
				'od_wait': 12,
				'max_depth': 7,
				'l2_leaf_reg': 1.75,
				'min_data_in_leaf': 25,
				'random_strength': 0.1,
				'max_bin': 100,
				'verbose': 0,
				'use_best_model': True
			}),
			"CB2C": CBC(**{'task_type': "GPU" if self.gpu_switch == "ON" else "CPU",
						   'objective': 'Logloss',
						   'eval_metric': "AUC",
						   'bagging_temperature': 0.5,
						   'colsample_bylevel': 0.50,
						   'iterations': 2500,
						   'learning_rate': 0.04,
						   'od_wait': 24,
						   'max_depth': 8,
						   'l2_leaf_reg': 1.235,
						   'min_data_in_leaf': 25,
						   'random_strength': 0.35,
						   'max_bin': 160,
						   'verbose': 0,
						   'use_best_model': True,
						   }
						),
			"CB3C": CBC(**{'task_type': "GPU" if self.gpu_switch == "ON" else "CPU",
						   'objective': 'Logloss',
						   'eval_metric': "AUC",
						   'bagging_temperature': 0.2,
						   'colsample_bylevel': 0.85,
						   'iterations': 2500,
						   'learning_rate': 0.025,
						   'od_wait': 10,
						   'max_depth': 7,
						   'l2_leaf_reg': 1.235,
						   'min_data_in_leaf': 8,
						   'random_strength': 0.60,
						   'max_bin': 160,
						   'verbose': 0,
						   'use_best_model': True,
						   }
						),
			"CB4C": CBC(**{'task_type': "GPU" if self.gpu_switch == "ON" else "CPU",
						   'objective': 'Logloss',
						   'eval_metric': "AUC",
						   'grow_policy': 'Lossguide',
						   'colsample_bylevel': 0.25,
						   'iterations': 2500,
						   'learning_rate': 0.035,
						   'od_wait': 24,
						   'max_depth': 7,
						   'l2_leaf_reg': 1.80,
						   'random_strength': 0.60,
						   'max_bin': 160,
						   'verbose': 0,
						   'use_best_model': True,
						   }
						),
			'HGB1C': HGBC(
				loss='log_loss',
				learning_rate=0.06,
				max_iter=800,
				max_depth=6,
				min_samples_leaf=12,
				l2_regularization=1.15,
				validation_fraction=0.1,
				n_iter_no_change=self.nbrnd_erly_stp,
				random_state=self.state
			),
			'HGB2C': HGBC(
				loss='log_loss',
				learning_rate=0.035,
				max_iter=700,
				max_depth=7,
				min_samples_leaf=9,
				l2_regularization=1.75,
				validation_fraction=0.1,
				n_iter_no_change=self.nbrnd_erly_stp,
				random_state=self.state
			),
		}
		return self

	def ScoreMetric(self, ytrue, ypred):
		'''
		This is the metric function for the competition scoring
		:param ytrue:
		:param y_pred:
		:return:
		'''
		return roc_auc_score(ytrue, ypred)

	def ClbMetric(self, ytrue, ypred):
		'''
		This is the calibration metric
		:param ytrue:
		:param ypred:
		:return:
		'''
		return brier_score_loss(ytrue, ypred)

	def PostProcessPred(self, ypred):
		'''
		This is an optional post-processing method
		:param ypred:
		:return:
		'''
		return np.clip(ypred, a_min=0, a_max=1)

	def TrainMdl(self, target: str, test_preds_req: str = 'Y', save_models='N'):
		'''
		This method trains and infers from the model suite and
		returns the predictions and scores
        It optionally predicts the test set too, if desired by the user
        Source for dropped columns:-
        https://www.kaggle.com/competitions/playground-series-s4e3/discussion/482401
		:param target:
		:param test_preds_req:
		:param save_models:
		:return:
		'''
		# Initializing I-O :-
		X, y, Xt = self.Xtrain[self.sel_cols], \
				   self.ytrain.copy(deep=True), \
				   self.Xtest[self.sel_cols]
		cols_drop = [
			'Source', 'id', 'Sum_of_Luminosity', \
			'X_Perimeter', 'SigmoidOfAreas', \
			'Edges_X_Index', 'Y_Minimum', 'Y_Maximum'
		]
		ens = OptunaEnsembler()
		self.FtreImp = pd.DataFrame(
			columns=self.methods, \
			index=[c for c in self.sel_cols if c not in cols_drop] \
			).fillna(0)

		# Making CV folds:-
		for fold_nb, (train_idx, dev_idx) in tqdm(enumerate(self.cv.split(X, self.y_grp))):
			Xtr = X.iloc[train_idx].drop(columns=cols_drop, errors='ignore')
			Xdev = X.iloc[dev_idx].drop(columns=cols_drop, errors='ignore')
			ytr = y.loc[y.index.isin(Xtr.index)]
			ydev = y.loc[y.index.isin(Xdev.index)]
			# Initializing the OOF and test set predictions:-
			oof_preds = pd.DataFrame(columns=self.methods, index=Xdev.index)
			mdl_preds = pd.DataFrame(columns=self.methods, index=Xt.index)
			PrintColor(f"\n{' = ' * 5} Fold {fold_nb + 1} {' = ' * 5}\n")
		# Initializing models across methods:-
		for method in tqdm(self.methods):
			model = Pipeline(steps=[('M', self.Mdl_Master.get(method))])
			# Fitting the model:-
			if 'CB' in method:
				model.fit(Xtr, ytr, \
						  M__eval_set=[(Xdev, ydev)], \
						  M__verbose=0,
						  M__early_stopping_rounds=CFG.nbrnd_erly_stp)
			elif 'LGBM' in method:
				model.fit(Xtr, ytr, \
						  M__eval_set=[(Xdev, ydev)], \
						  M__callbacks=[log_evaluation(0), \
										early_stopping(stopping_rounds=CFG.nbrnd_erly_stp, verbose=False, ),
										]
						  )
			elif 'XGB' in method:
				model.fit(Xtr, ytr, \
						  M__eval_set=[(Xdev, ydev)], \
						  M__verbose=0, )
			else:
				model.fit(Xtr, ytr)
			# Collating feature importance:-
			try:
				self.FtreImp[method] += model['M'].feature_importances_
			except:
				pass
			# Collecting predictions and scores and post-processing OOF based on model method:-
			dev_preds = model.predict_proba(Xdev)[:, 1]
			train_preds = model.predict_proba(Xtr)[:, 1]
			tr_score = self.ScoreMetric(ytr.values.flatten(), \
										train_preds)
			score = self.ScoreMetric(ydev.values.flatten(), dev_preds)
			PrintColor(f'OOF={score:.5f} | train = {tr_score:.5f} | {method}', color=Fore.CYAN)
			oof_preds[method] = dev_preds
			# Integrating the predictions and scores:-
			self.Scores.at[fold_nb, method] = np.round(score, decimals=6)
			self.TrainScores.at[fold_nb, method] = np.round(tr_score, decimals=6)
			if test_preds_req == 'Y':
				mdl_preds[method] = self.PostProcessPred(model.predict_proba(Xt.drop(columns=cols_drop,
																					 errors='ignore')))
		try:
			del dev_preds, train_preds, tr_score, score
		except:
			pass
		# Ensembling the predictions:-
		oof_preds['Ensemble'] = ens.fit_predict(ydev, oof_preds[self.method])
		score = self.ScoreMetric(ydev, oof_preds['Ensemble'].values)
		self.OOF_Preds = pd.concat([self.OOF_Preds, oof_preds], axis=0, ignore_index=False)
		self.Scores.at[fold_nb, 'Ensemble'] = np.round(score, 6)
		if test_preds_req == 'Y':
			mdl_preds['ensemble'] = ens.predict(mdl_preds[self.methods])
			self.Mdl_Preds = pd.concat([self.Mdl_Preds, mdl_preds], axis=1, ignore_index=False)
		# Averaging the predictions afeter all folds:-
		self.OOF_Preds = self.OOF_Preds.groupby(level=0).mean()
		if test_preds_req == 'Y':
			self.Mdl_Preds = self.Mdl_Preds[self.methods + ['Ensemble']].groupby(level=0).mean()
		return self.OOF_Preds, self.Mdl_Preds, self.Scores, self.TrainScores

	def MakePseudoLbl(self, up_cutoff: float, low_cutoff: float, **kwargs):
		"""
		This method makes pseudo-labels using confident test set predictions to add to the training data
		:param up_cutoff:
		:param low_cutoff:
		:param kwargs:
		:return:
		"""
		df = self.Mdl_Preds.loc[ \
			(self.Mdl_Preds.Ensemble >= up_cutoff) | (self.Mdl_Preds.Ensemble <= low_cutoff), \
			'Ensemble'
		]
		PrintColor(f'-->Pseudo Label additions form test set={df.shape[0]:,.0f}', \
				   color=Fore.RED)
		df = df.astype(np.uint8)
		new_ytrain = pd.concat([self.ytrain, df], axis=0, ignore_index=True)
		new_ytrain.index = range(len(new_ytrain))

		new_Xtrain = pd.concat([self.Xtrain, self.Xtest.loc[df.index]], \
							   axis=0, \
							   ignore_index=True)
		new_Xtrain.index = range(len(new_Xtrain))
		#  Verifying the additions:-
		PrintColor(f"---> Revised train set shapes after pseudo labels = {new_Xtrain.shape} {new_ytrain.shape}");
		return new_Xtrain, new_ytrain

	def MakeMLPlots(self):
		'''
		This method makes plots for the ML models,
		 including feature importance and calibration curves
		:return:
		'''
		fig, axes = plt.subplots(len(self.methods), \
								 2, \
								 figsize=(35, len(self.methods) * 10), \
								 gridspec_kw={'hspace': 0.6, 'wspace': 0.2}, \
								 width_ratios=[0.75, 0.25], \
								 );
		for i, col in enumerate(self.methods):
			try:
				ax = axes[i, 0];
			except:
				ax = axes[0];
			self.FtreImp[col].plot.bar(ax=ax, color='#0073e6');
			ax.set_title(f"{col} Importances", **CFG.title_specs);
			ax.set(xlabel='', ylabel='');
			try:
				ax = axes[i, 1];
			except:
				ax = axes[1];
			Clb.from_predictions(self.ytrain[0:len(self.OOF_Preds)],
								 self.OOF_Preds[col],
								 n_bins=20,
								 ref_line=True,
								 **{'color': '#0073e6', 'linewidth': 1.2,
									'markersize': 3.75, 'marker': 'o', 'markerfacecolor': '#cc7a00'},
								 ax=ax
								 )
			ax.set_title(f"{col} Calibration", **CFG.title_specs);
			ax.set(xlabel='', ylabel='', );
			ax.set_yticks(np.arange(0, 1.01, 0.05),
						  labels=np.round(np.arange(0, 1.01, 0.05), 2), fontsize=7.0);
			ax.set_xticks(np.arange(0, 1.01, 0.05),
						  labels=np.round(np.arange(0, 1.01, 0.05), 2),
						  fontsize=6.25,
						  rotation=90
						  );
			ax.legend('');
		plt.tight_layout();
		plt.show();


print();
collect()


class Utils:
	'''
	    This class plots the final scores and generates adjutant model utilities
	'''

	def __init__(self, targets):
		self.targets = targets

	def DisplayAdjTbl(self, *args):
		'''
		This function displays pandas tables in an adjacent manner, sourced from the below link-
        https://stackoverflow.com/questions/38783027/jupyter-notebook-display-two-pandas-tables-side-by-side
		:param args:
		:return:
		'''
		html_str = ''
		for df in args:
			html_str += df.to_html()
		display_html(html_str.replace('table', 'table style="display:inline"'), raw=True);
		collect();

	def DisplayScores(self, Scores: pd.DataFrame, TrainScores: pd.DataFrame):
		'''
		This method displays the scores and their means
		:param Scores:
		:param TrainScores:
		:return:
		'''
		methods = Scores.columns[0:-2];
		args = \
			[Scores.style.format(precision=5). \
				 background_gradient(cmap="Pastel2", subset=methods). \
				 set_caption(f"\nOOF scores across methods and folds\n"),

			 TrainScores.style.format(precision=5). \
				 background_gradient(cmap="Pastel2", subset=methods). \
				 set_caption(f"\nTrain scores across methods and folds\n")
			 ];
		PrintColor(f"\n\n\n---> OOF score across all methods and folds\n", color=Fore.LIGHTMAGENTA_EX);
		self.DisplayAdjTbl(*args);

		print('\n');
		# display(Scores.groupby("Target")[["Ensemble"]].mean(). \
		# 		transpose(). \
		# 		style.format(precision=5). \
		# 		background_gradient(cmap="Spectral", axis=1, subset=self.targets). \
		# 		set_caption(f"\nOOF mean scores across methods and folds\n")
		# 		)

		PrintColor(f"\n---> Mean ensemble score OOF = {np.mean(Scores['Ensemble']):.5f}\n");


collect()
print()

if CFG.ML == 'Y':
	sel_cols = Xtrain.columns
	PrintColor(f'\n ---> Selected model columns')
	cat_ftre = list(set(pp.cat_cols))
	with np.printoptions(linewidth=150):
		PrintColor(f'\n--> All Selected columns\n')
		pprint(np.array(sel_cols))

		PrintColor(f'\n--> All category columns\n')
		pprint(np.array(cat_ftre))

if CFG.ML == 'Y':
	OOF_Preds, Mdl_Preds, Scores, TrainScores = pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
	for target in CFG.targets:
		md = MdlDeveloper(Xtrain, \
						  ytrain[target], \
						  ytrain[target], \
						  Xtest, \
						  sel_cols=sel_cols, \
						  cat_cols=cat_ftre, \
						  enc_cols=[] \
						  )
		oof_preds, mdl_preds, scores, trainscores = md.TrainMdl(
			test_preds_req='Y', \
			target=target)
		OOF_Preds = pd.concat([oof_preds.assign(Target=target), OOF_Preds], \
							  axis=0, \
							  ignore_index=False)
		Mdl_Preds = pd.concat([mdl_preds.assign(Target=target), Mdl_Preds], \
							  axis=0, \
							  ignore_index=False)
		Scores = pd.concat([scores.assign(Target=target), Scores], \
						   axis=0, \
						   ignore_index=True)
		TrainScores = pd.concat([trainscores.assign(Target=target), TrainScores], \
								axis=0, \
								ignore_index=True)
	clear_output()
	utils = Utils(CFG.targets)
	utils.DisplayScores(Scores, TrainScores)

print()
collect()
# ---> OOF score across all methods and folds

if CFG.ML == 'Y':
	for col in CFG.targets:
		pp.sub_f1[col] = 1 - Mdl_Preds.loc[Mdl_Preds.Target == col, 'Ensemble '].values
	sub1 = pd.read_csv(f'../input/playgrounds4e03ancillary/89652_submission.csv')[CFG.targets]
	pp.sub_f1[CFG.targets] = pp.sub_f1[CFG.targets].values * 0.1 + sub1 * 0.9

	pp.sub_f1.to_csv(f'Submission_V{CFG.version_nb}.csv', index=False)
	OOF_Preds.to_csv(f'OOF_Preds_V{CFG.version_nb}.csv', index=False)
	Mdl_Preds.to_csv(f'Mdl_Preds_V{CFG.version_nb}.csv', index=False)
	# display(pp.sub_fl.head(10).style.set_caption(f"\nSubmission file\n").format(precision=3));
print()
collect()
