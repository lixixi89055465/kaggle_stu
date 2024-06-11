# -*- coding: utf-8 -*-
# @Time : 2024/6/3 11:19
# @Author : nanji
# @Site : https://www.kaggle.com/code/gauravduttakiit/pss4e6-flaml-roc-auc-ovo
# @File : solve02.py
# @Software: PyCharm 
# @Comment :
from sklearn.pipeline import Pipeline, make_pipeline
from tqdm.notebook import tqdm
import numpy as np
import pandas as pd
from datetime import datetime
import optuna
from optuna import Trial, trial, create_study
from optuna.pruners import HyperbandPruner
from optuna.samplers import TPESampler, CmaEsSampler
from scipy import stats

optuna.logging.set_verbosity = optuna.logging.ERROR
from functools import partial

from sklearn.metrics import accuracy_score, roc_auc_score, make_scorer
from sklearn.model_selection import (RepeatedStratifiedKFold as RSKF, \
									 StratifiedKFold as SKF, \
									 StratifiedGroupKFold as SGKF, \
									 KFold, \
									 RepeatedKFold as RKF, \
									 cross_val_score, \
									 cross_val_predict)
# ML Model training : ~
from sklearn.metrics import accuracy_score, roc_auc_score, make_scorer
from xgboost import DMatrix, XGBClassifier as XGBC
from lightgbm import log_evaluation, early_stopping, LGBMClassifier as LGBMC
from catboost import CatBoostClassifier as CBC, Pool
from sklearn.ensemble import HistGradientBoostingClassifier as HGBC, \
	RandomForestClassifier as RFC
from sklearn.metrics import brier_score_loss

train = pd.read_csv('../input/playground-series-s4e6/train.csv')
# print(train.head())
test = pd.read_csv('../input/playground-series-s4e6/test.csv')
# print(test.head())
print('0' * 100)
# print(train.info())
print('1' * 100)
# print(train.nunique())

r1 = round(train['Target'].value_counts() * 100 / len(train), 2)
print(r1)
print('2' * 100)
print(train.isnull().sum())
import re

train = train.rename(columns=lambda x: re.sub('[^A-Za-z0-9_]+', '', x))
print('3' * 100)
print(train.head())

test = test.rename(columns=lambda x: re.sub('[^A-Za-z0-9_]+', '', x))
print(test.head())


def reduce_mem_usage(df):
	""" iterate through all the columns of a dataframe and modify the data type
	    to reduce memory usage.
	"""
	start_mem = df.memory_usage().sum() / 1024 ** 2
	print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))
	for col in df.columns:
		col_type = df[col].dtype
		if col_type != object:
			c_min = df[col].min()
			c_max = df[col].max()
			if str(col_type)[:3] == 'int':
				if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
					df[col] = df[col].astype(np.int8)
				elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
					df[col] = df[col].astype(np.int16)
				elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
					df[col] = df[col].astype(np.int32)
				elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
					df[col] = df[col].astype(np.int64)
			else:
				if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
					df[col] = df[col].astype(np.float16)
				elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
					df[col] = df[col].astype(np.float32)
				else:
					df[col] = df[col].astype(np.float64)
		else:
			df[col] = df[col].astype('object')
	end_mem = df.memory_usage().sum() / 1024 ** 2
	print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
	print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
	return df


train = reduce_mem_usage(train)
test = reduce_mem_usage(test)


class CFG:
	"""
	Configuration class for parameters and CV strategy for tuning and training
	Some parameters may be unused here as this is a general configuration class
	""";
	# Data preparation
	version_nb = "bc"
	test_req = 'Y'
	test_sample_frac = 0.025
	gpu_switch = 'OFF'
	state = 42
	targets = ['Target']
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
		return accuracy_score(ytrue, ypred)

	def _objective(self, trial, y_true, y_preds):
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
		# weighted_pred = np.average(np.array(y_preds), axis=axis, weights=weights)
		weighted_pred = stats.mode(y_preds, axis=1)[0]
		score = self.ScoreMetric(y_true.values.flatten(), weighted_pred.reshape(-1))
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
			sampler=TPESampler(seed=self.random_state),
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

	def predict(self, y_preds):
		'''
		This method predicts using the fitted Optuna objective;
		:param y_preds:
		:return:
		'''
		assert self.weights is not None, 'OptunaWeights error, must be fitted before predict';
		if isinstance(y_preds, list):
			weighted_pred = np.average(np.array(y_preds), axis=0, weights=self.weights)
		elif len(y_preds.shape) == 1:
			weighted_pred = y_preds
		else:
			# weighted_pred = np.average(np.array(y_preds).reshape(-1, 1), axis=1, weights=self.weights)
			weighted_pred = stats.mode(y_preds, axis=1)[0]

		return weighted_pred

	def fit_predict(self, y_true, y_pres):
		self.fit(y_true, y_pres)
		return self.predict(y_pres)

	def weights(self):
		return self.weights


# Machine Learning Algorithm (MLA) Selection and Initialization
from sklearn import ensemble, gaussian_process, \
	linear_model, naive_bayes, \
	neighbors, svm, tree, discriminant_analysis
from xgboost import XGBClassifier


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
		self.testId = Xtest['id']
		self.Xtest = Xtest.drop('id', axis=1)
		self.Xtest_R = {}
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
			# 'ada': ensemble.AdaBoostClassifier(),# Score: 0.82052
			# 'bc': ensemble.BaggingClassifier(n_jobs=-1),#
			'etc': ensemble.ExtraTreesClassifier(n_jobs=-1),
			# 'gbc': ensemble.GradientBoostingClassifier(),
			# 'rfc': ensemble.RandomForestClassifier(n_jobs=-1),
			## Gaussian Processes: http://scikit-learn.org/stable/modules/gaussian_process.html#gaussian-process-classification-gpc
			### 'gpc': gaussian_process.GaussianProcessClassifier(n_jobs=3),
			## GLM: http://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
			# 'lr': linear_model.LogisticRegressionCV(n_jobs=-1),
			# # Navies Bayes: http://scikit-learn.org/stable/modules/naive_bayes.html
			# 'bnb': naive_bayes.BernoulliNB(),
			# 'gnb': naive_bayes.GaussianNB(),
			# # Nearest Neighbor: http://scikit-learn.org/stable/modules/neighbors.html
			# 'knn': neighbors.KNeighborsClassifier(n_jobs=-1),
			# # SVM: http://scikit-learn.org/stable/modules/svm.html
			### 'svc': svm.SVC(probability=True),
			# # xgboost: http://xgboost.readthedocs.io/en/latest/model.html
			# 'xgb': XGBClassifier()
		}
		return self

	def ScoreMetric(self, ytrue, ypred):
		'''
		This is the metric function for the competition scoring
		:param ytrue:
		:param y_pred:
		:return:
		'''
		return accuracy_score(ytrue, ypred)

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
				   self.ytrain.copy(), \
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
			# PrintColor(f"\n{' = ' * 5} Fold {fold_nb + 1} {' = ' * 5}\n")
			# Initializing models across methods:-
			for method in tqdm(self.methods):
				print(f'{datetime.now()}; fold: {fold_nb} ; method :{method}  start ')
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
					model.fit(Xtr, ytr.values.reshape(-1))
				# Collating feature importance:-
				try:
					self.FtreImp[method] += model['M'].feature_importances_
				except:
					pass
				# Collecting predictions and scores and post-processing OOF based on model method:-
				dev_preds = model.predict(Xdev)
				# dev_preds = model.predict_proba(Xdev)[:, 1]
				# train_preds = model.predict_proba(Xtr)[:, 1]
				train_preds = model.predict(Xtr)
				tr_score = self.ScoreMetric(ytr.values.flatten(), \
											train_preds)
				score = self.ScoreMetric(ydev.values.flatten(), dev_preds)
				# PrintColor(f'OOF={score:.5f} | train = {tr_score:.5f} | {method}', color=Fore.CYAN)
				oof_preds[method] = dev_preds  # TODO importance
				# Integrating the predictions and scores:-
				self.Scores.at[fold_nb, method] = np.round(score, decimals=6)
				self.TrainScores.at[fold_nb, method] = np.round(tr_score, decimals=6)
				if test_preds_req == 'Y':
					mdl_preds[method] = self.PostProcessPred(model.predict_proba(Xt.drop(columns=cols_drop,
																						 errors='ignore')))
				print(f'{datetime.now()}; fold: {fold_nb} ; method :{method}  end ')
			try:
				del dev_preds, train_preds, tr_score, score
			except:
				pass
			# Ensembling the predictions:-
			oof_preds['Ensemble'] = ens.fit_predict(ydev, oof_preds[self.methods])
			score = self.ScoreMetric(ydev, oof_preds['Ensemble'].values)
			self.OOF_Preds = pd.concat([self.OOF_Preds, oof_preds], axis=0, ignore_index=False)
			self.Scores.at[fold_nb, 'Ensemble'] = np.round(score, 6)
			if test_preds_req == 'Y':
				mdl_preds['Ensemble'] = ens.predict(mdl_preds[method])
				self.Mdl_Preds = pd.concat([self.Mdl_Preds, mdl_preds], axis=1, ignore_index=False)

		# Averaging the predictions afeter all folds:-
		self.OOF_Preds = self.OOF_Preds.groupby(level=0).mean()
		result = pd.DataFrame(columns=self.methods, index=Xt.index)
		if test_preds_req == 'Y':
			self.Mdl_Preds = self.Mdl_Preds[self.methods + ['Ensemble']].groupby(level=0).mean()
			for key, model in self.Mdl_Master.items():
				result[key] = model.predict(self.Xtest)
			result['Ensemble'] = ens.predict(result)

		return self.OOF_Preds['Ensemble'].astype(np.int8), self.Mdl_Preds, self.Scores, self.TrainScores, result[
			'Ensemble'].astype(np.int8)

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
		# PrintColor(f'-->Pseudo Label additions form test set={df.shape[0]:,.0f}', color=Fore.RED)
		df = df.astype(np.uint8)
		new_ytrain = pd.concat([self.ytrain, df], axis=0, ignore_index=True)
		new_ytrain.index = range(len(new_ytrain))

		new_Xtrain = pd.concat([self.Xtrain, self.Xtest.loc[df.index]], \
							   axis=0, \
							   ignore_index=True)
		new_Xtrain.index = range(len(new_Xtrain))
		#  Verifying the additions:-
		# PrintColor(f"---> Revised train set shapes after pseudo labels = {new_Xtrain.shape} {new_ytrain.shape}");
		return new_Xtrain, new_ytrain

	def MakeMLPlots(self):
		'''
		This method makes plots for the ML models,
		 including feature importance and calibration curves
		:return:
		'''
		pass


from sklearn.preprocessing import LabelEncoder

# pp = Preprocessor();
# pp.DoPreprocessing();
target = 'Target'
cols_drop = ['id', 'Target']
ytrain = train[CFG.targets]
labelEncoder = LabelEncoder()
ytrain = labelEncoder.fit_transform(ytrain)
ytrain = pd.DataFrame(ytrain).astype(np.int8)

train = train.drop(cols_drop, axis=1)
sel_cols = train.columns
print('2' * 100)
print(train.nunique())
featureCount = train.nunique()
cat_ftre = featureCount.loc[featureCount <= 10].index.to_list()
cont_cols = [c for c in featureCount.index if c not in cat_ftre]

if CFG.ML == 'Y':
	OOF_Preds, Mdl_Preds, Scores, TrainScores = pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
	md = MdlDeveloper(train, ytrain, ytrain, test, sel_cols=sel_cols, cat_cols=cat_ftre, enc_cols=[])
	oof_preds, mdl_preds, scores, trainscores, result = md.TrainMdl(test_preds_req='Y', target=target)
	print(oof_preds)
	# OOF_Preds = pd.concat([oof_preds.assign(Target=target), OOF_Preds], \
	# 					  axis=0, \
	# 					  ignore_index=False)
	OOF_Preds = pd.DataFrame({"id": test.id, "Target": labelEncoder.inverse_transform(result)})
	Mdl_Preds = pd.concat([mdl_preds.assign(Target=target), Mdl_Preds], \
						  axis=0, \
						  ignore_index=False)
	Scores = pd.concat([scores.assign(Target=target), Scores], \
					   axis=0, \
					   ignore_index=True)
	TrainScores = pd.concat([trainscores.assign(Target=target), TrainScores], \
							axis=0, \
							ignore_index=True)

	name = ''
	for item in md.Mdl_Master.items():
		name = name + "_" + item.key()
	# TODO  6-5
	OOF_Preds.to_csv(f'Submission_V_{name}.csv', index=False)

print(f'{datetime.now()} end !!!!!')
