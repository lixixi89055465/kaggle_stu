# -*- coding: utf-8 -*-
# @Time : 2024/3/23 下午4:57
# @Author : nanji
# @Site :
# @File : solve01.py
# @Software: PyCharm 
# @Comment : https://www.kaggle.com/code/getanmolgupta01/defect-pred-eda-xgboost-lgbm-catboost

'''

# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load
'''

import warnings
warnings.filterwarnings('ignore')
import numpy as np  # linear algebra
import pandas as pd  # input processing, CSV file I/O (e.g. pd.read_csv)

# Input input files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory


import os
# for dirname, _, filenames in os.walk('../input/'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))


# Libray for Data Manipulation.
import pandas as pd
import numpy as np

# Library for Data Visualization.
import seaborn as sns
import matplotlib.pyplot as plt

# sns.set(style="white", font_scale=1.5)
# sns.set(rc={"axes.facecolor": "#FFFAF0", "figure.facecolor": "#FFFAF0"})
# sns.set_context("poster", font_scale=.7)
import warnings

warnings.filterwarnings('ignore')

# Library to perform Statistical Analysis.
from scipy import stats
from scipy.stats import chi2
from scipy.stats import chi2_contingency

# Library to Display whole Dataset.
pd.set_option("display.max.columns", 100)

train_df = pd.read_csv('../input/train.csv')
test_df = pd.read_csv('../input/test.csv')
print(train_df.shape)
print(test_df.shape)
print(train_df.columns)

print("1" * 100)
print(train_df.info())

# Identify the input types of columns
column_data_types = train_df.dtypes
# print("2"*100)
print(column_data_types)

# Count the numerical and categorical columns
numerical_count = 0
categorical_count = 0

for column_name, data_type in column_data_types.items():
	if np.issubdtype(data_type, np.number):
		numerical_count += 1
	else:
		categorical_count += 1

# Print the counts
print(f"There are {numerical_count} Numerical Columns in dataset")
print(f"There are {categorical_count} Categorical Columns in dataset")

# 5. Checking if There's Any Duplicate Records.¶
print("Duplicates in Dataset: ", train_df.duplicated().sum())

# There are no duplicate records present in the dataset.
# 6. Computing Total No. of Missing Values and the Percentage of Missing Values¶
missing_data = train_df.isnull().sum().to_frame().rename(
	columns={0: "Total No. of Missing Values"}
)
missing_data['% of Missing Values'] = round((
													missing_data['Total No. of Missing Values'] / len(train_df)) * 100,
											2)
print('0' * 100)
print(missing_data)
# None of the Attribute are having Missing Values.
print('1' * 100)
# print(round(train_df.describe().T, 2))
print(round(test_df.describe().T, 2))
# The Minimum Pixels_Areas is 4 which conveys that some steel sheets are too small.
# 8. Dropping Attritbutes which doesn't imply any meaningful insights in our analysis.¶

cols = ["id"]
train_df.drop(columns=cols, inplace=True)
test_df.drop(columns=cols, inplace=True)
# creating a 'Fault_type column for EDA purpose '
train_df['Fault_Type'] = train_df[
	['Pastry', 'Z_Scratch', 'K_Scatch', 'Stains', 'Dirtiness', 'Bumps', 'Other_Faults']
] \
	.idxmax(axis=1)
print('2' * 100)

# # 1. Visualizing the Distribution of each Fault_Type class¶
# plt.figure(figsize=(17, 6))
# plt.subplot(1, 2, 1)
# Fault_Type_counts = train_df["Fault_Type"].value_counts()
# sns.barplot(x=Fault_Type_counts.index, y=Fault_Type_counts.values, palette='Set2')
# plt.title("Distribution of Fault_Type Classes", fontweight="black", size=14, pad=15)
# for i, v in enumerate(Fault_Type_counts.values):
#     plt.text(i, v, v, ha="center", fontsize=14)
#
# plt.xticks(rotation=90)
# # Visualization to show distribution of Fault_Type classes in percentage
#
# plt.subplot(1, 2, 2)
# colors = sns.color_palette('Set2', len(Fault_Type_counts))
# plt.pie(Fault_Type_counts, labels=Fault_Type_counts.index, autopct="%.2f%%", textprops={"size": 14},
#         colors=colors, startangle=90)
# center_circle = plt.Circle((0, 0), 0.3, fc='white')
# fig = plt.gcf()
# fig.gca().add_artist(center_circle)
# plt.title("Distribution of Fault_Type Classes", fontweight="black", size=14, pad=15)
# plt.show()

#  Inference:
# Pastry have 11.88% distribution
# Z_Scratch have 5.98% distribution
# K_Scatch have 17.85% distribution
# Stains have 2.96% distribution
# Dirtiness have 2.52% distribution
# Bumps have 24.77% distribution
# Other_Faults have 34.03% distribution
# 2. Visualising the distribution of each features¶
numerical_features = ['X_Minimum', 'X_Maximum', 'Y_Minimum', 'Y_Maximum', 'Pixels_Areas',
					  'X_Perimeter', 'Y_Perimeter', 'Sum_of_Luminosity', 'Minimum_of_Luminosity',
					  'Maximum_of_Luminosity', 'Length_of_Conveyer', 'TypeOfSteel_A300',
					  'TypeOfSteel_A400', 'Steel_Plate_Thickness', 'Edges_Index', 'Empty_Index',
					  'Square_Index', 'Outside_X_Index', 'Edges_X_Index', 'Edges_Y_Index',
					  'Outside_Global_Index', 'LogOfAreas', 'Log_X_Index', 'Log_Y_Index',
					  'Orientation_Index', 'Luminosity_Index', 'SigmoidOfAreas']

# Set the figure size and arrange plots horizontally in pairs
# fig, axes = plt.subplots(nrows=(len(numerical_features) + 2) // 3, ncols=3, figsize=(30, 40))
# # Flatten the axes array for easy indexing
# axes = axes.flatten()

# Loop through the selected columns and create histograms with density
# for i, col in enumerate(numerical_features):
#     sns.histplot(data=train_df, x=col, hue='Fault_Type', \
# 				 multiple="stack", bins=20, \
# 				 kde=True, palette='viridis',\
# 				 ax=axes[i])
#
#     axes[i].set_title(f'Histogram with Density for {col}')
#     axes[i].set_xlabel(col)
#     axes[i].set_ylabel('Density')


# Remove any empty subplots if the number of columns is odd
# if len(numerical_features) % 3 != 0:
#     for j in range(len(numerical_features) % 3, 3):
#         fig.delaxes(axes[-j - 1])
#
# plt.tight_layout()
# plt.show()
'''
From these plots we can make the following conclusions about the features' distributions:

Normal: Maximum_of_Luminosity, Empty_Index, Square_Index, and Luminosity_Index
Close to normal: Minimum_of_Luminosity, and Orientation_Index
Skewed to the right: Y_Minimum, Y_Maximum, Pixels_Areas, X_Perimeter, Y_Perimeter, and Sum_of_Luminosity
Close to skewed to the right: Log_Y_Index
Close to skewed to the left: Edges_X_Index
Close to uniform: X_Minimum, X_Maximum, EdgesIndex, Edges_Y_Index, and SigmoidOfAreas
The features not mentioned have distributions that we are unable to categorise.
We will later perform transformations on the features that fell into the categories: 
Close to normal, Skewed to the right, Close to skewed to the right, 
and Close to skewed to the left. 
We aim to transform the values in such a way that their distributions become more normal.¶
'''


def transform(X):
	eps = 1e-5
	X['Sum_of_Luminosity'] = np.log(X['Sum_of_Luminosity'] + eps)
	X['Pixels_Areas'] = np.log(X['Pixels_Areas'] + eps)
	X['X_Perimeter'] = np.log(X['X_Perimeter'] + eps)
	X['Steel_Plate_Thickness'] = np.log(X['Steel_Plate_Thickness'] + eps)
	X['Y_Perimeter'] = np.log(X['Y_Perimeter'] + eps)
	X['Outside_X_Index'] = np.log(X['Outside_X_Index'] + eps)
	X['Y_Minimum'] = np.log(X['Y_Minimum'] + eps)
	X['Y_Maximum'] = np.log(X['Y_Maximum'] + eps)
	return X


a = transform(train_df)
print(a.shape)
a = transform(test_df)
print(a.shape)
# Featutre Engineering and data preparation
df = train_df.copy()
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm

df = sm.add_constant(df.select_dtypes(include=[np.number]))
vif = pd.DataFrame()
vif["Variable"] = df.columns
vif["VIF"] = [variance_inflation_factor(df.values, i) for i in range(df.shape[1])]
print('1' * 100)
# print(vif)

highly_correlated_variable = []
for index, row in vif.iterrows():
	if row['VIF'] > 6.5:
		highly_correlated_variable.append(row['Variable'])
highly_correlated_variable.remove('const')
print('2' * 100)
print(highly_correlated_variable)

import numpy as np

epsilon = 1e-5


def generate_features(X):
	# Location Features
	X['X_Distance'] = X['X_Maximum'] - X['X_Minimum']
	X['Y_Distance'] = X['Y_Maximum'] - X['Y_Minimum']
	# Density Feature
	# X['Density'] = X['Pixels_Areas'] / (X['X_Perimeter'] + X['Y_Perimeter'])
	# Relative Perimeter Feature
	X['Relative_Perimeter'] = X['X_Perimeter'] / (X['X_Perimeter'] + X['Y_Perimeter'] + epsilon)
	# Circularity Feature
	X['Circularity'] = X['Pixels_Areas'] / (X['X_Perimeter'] ** 2)
	# Symmetry Index Feature
	# X['Symmetry_Index'] = np.abs(X['X_Distance'] - X['Y_Distance']) / (X['X_Distance'] + X['Y_Distance'] + epsilon)
	# Color Contrast Feature
	X['Color_Contrast'] = X['Maximum_of_Luminosity'] - X['Minimum_of_Luminosity']
	# Combined Geometric Index Feature
	# X['Combined_Geometric_Index'] = X['Edges_Index'] * X['Square_Index']

	# Interaction Term Feature
	# X['X_Distance*Pixels_Areas'] = X['X_Distance'] * X['Pixels_Areas']

	# Additional Features
	# X['sin_orientation'] = np.sin(X['Orientation_Index'])
	# X['Edges_Index2'] = np.exp(X['Edges_Index'] + epsilon)
	# X['X_Maximum2'] = np.sin(X['X_Maximum'])
	# X['Y_Minimum2'] = np.sin(X['Y_Minimum'])
	# X['Aspect_Ratio_Pixels'] = np.where(X['Y_Perimeter'] == 0, 0, X['X_Perimeter'] / X['Y_Perimeter'])
	# X['Aspect_Ratio'] = np.where(X['Y_Distance'] == 0, 0, X['X_Distance'] / X['Y_Distance'])

	# Average Luminosity Feature
	# X['Average_Luminosity'] = (X['Sum_of_Luminosity'] + X['Minimum_of_Luminosity']) / 2

	# Normalized Steel Thickness Feature
	# X['Normalized_Steel_Thickness'] = (X['Steel_Plate_Thickness'] - X['Steel_Plate_Thickness'].min())
	# 				/ (X['Steel_Plate_Thickness'].max() - X['Steel_Plate_Thickness'].min())

	# Logarithmic Features
	# X['Log_Perimeter'] = np.log(X['X_Perimeter'] + X['Y_Perimeter'] + epsilon)
	# X['Log_Luminosity'] = np.log(X['Sum_of_Luminosity'] + epsilon)
	# X['Log_Aspect_Ratio'] = np.log(X['Aspect_Ratio'] ** 2 + epsilon)

	# Statistical Features
	X['Combined_Index'] = X['Orientation_Index'] * X['Orientation_Index']
	X['Sigmoid_Areas'] = 1 / (1 + np.exp(-X['LogOfAreas'] + epsilon))
	return X


# print('3'*100)
a=generate_features(train_df)
print(a.shape)
print('4'*100)
b=generate_features(test_df)
print(b.shape)

train_df = train_df.drop(highly_correlated_variable, axis=1)
test_df = test_df.drop(highly_correlated_variable, axis=1)

from sklearn.preprocessing import LabelEncoder

label = LabelEncoder()
train_df['Fault_Type'] = label.fit_transform(train_df['Fault_Type'])
label_mapping = dict(zip(label.classes_, label.transform(label.classes_)))
print('Label Mapping:')
print(label_mapping)
# plt.figure(figsize=(40,20))
# plt.title('Correlation Plot')
# sns.heatmap(
# 	train_df.corr(),
# 	linewidths=5,
# 	annot=True,
# 	square=True,
# 	annot_kws={'size':10,},
# 	cmap='YlGnBu'
# )
# plt.show()

# Calculate the correlation matrix
correlation_matrix = train_df.corr()
# Create a mask to identify the features with a correlation coefficient greater than or equal to 0.75
high_correlation_mask = correlation_matrix >= 0.75
# Identify and list the highly correlated features
highly_correlated_features = []
for feature in high_correlation_mask.columns:
	correlated_with = high_correlation_mask.index[high_correlation_mask[feature]].tolist()
	for correlated_feature in correlated_with:
		if feature != correlated_feature and (correlated_feature, feature) not in highly_correlated_features:
			highly_correlated_features.append((feature, correlated_feature))

# Print the highly correlated features
print("Highly correlated features:")
for feature1, feature2 in highly_correlated_features:
	print(f'{feature1} and {feature2}')
# Highly correlated features:

print('3' * 100)
print(train_df.cov()['Fault_Type'])
y = train_df['Fault_Type']
y1 = train_df['Pastry']
y2 = train_df['Z_Scratch']
y3 = train_df['K_Scatch']
y4 = train_df['Stains']
y5 = train_df['Dirtiness']
y6 = train_df['Bumps']
y7 = train_df['Other_Faults']
x = train_df.drop(['Pastry', 'Z_Scratch', \
				   'K_Scatch', 'Stains', \
				   'Dirtiness', 'Bumps', \
				   'Other_Faults', 'Fault_Type'], axis=1)

# Computing Class Weights¶
from sklearn.utils.class_weight import compute_class_weight
# Convert y to a NumPy array if it's not already one
arr = np.array(y)

# Calculate unique classes in y
unique_classes = np.unique(arr)

# Convert unique_classes to a list to ensure hashability
unique_classes_list = list(unique_classes)

# Calculate class weights based on the training data
class_weights = compute_class_weight('balanced', 
                                     classes=unique_classes_list,
                                     y=y)

# Create a dictionary of class weights
class_weights_dict = dict(zip(unique_classes_list, class_weights))
print(class_weights_dict)
class_weights_dict = {0: 0.5766795691181325, \
					  1: 5.660972017673049, \
					  2: 0.8002248407378107, \
					  3: 0.4198121450415028, \
					  4: 1.2020890668001, \
					  5: 4.833752515090543, \
					  6: 2.3874534161490684}

print(x.shape)
print('4' * 100)
print(test_df.shape)
print('5' * 100)
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

scaler = RobustScaler()
standardscl = StandardScaler()
minmax = MinMaxScaler()
x_scaled = scaler.fit_transform(x)
test_df_scaled = scaler.transform(test_df)

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = \
	train_test_split(x, y, test_size=0.2, random_state=42)
x_train_scaled, x_test_scaled, y_train, y_test = \
	train_test_split(x_scaled, y, test_size=0.2, random_state=42)
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

pca = PCA()
pca.fit(x_scaled)
explained_variance_ratio = pca.explained_variance_ratio_
cumulative_variance_ratio = explained_variance_ratio.cumsum()
# plt.plot(cumulative_variance_ratio)
# plt.xlabel('Number of Components')
# plt.ylabel('Cumulative Explained Variance')
# plt.show()
## PCA- Transformation
# pca=PCA(n_components=20)
# pca

# Scaled Data
# x_scaled=pca.fit_transform(x_scaled)
# x_train1 = pca.transform(x_train1)
# Scaled data
# test_df_scaled=pca.transform(test_df_scaled)
# x_test1 = pca.transform(x_test1)

# Machine learning algorithms
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier, \
	AdaBoostClassifier, GradientBoostingClassifier, \
	VotingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier, Pool
import tensorflow
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout
import os
from sklearn.base import ClassifierMixin
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.multiclass import OneVsRestClassifier
# from scikeras.wrappers import KerasClassifier

# for hypertuning

import optuna
from collections import Counter
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, RepeatedStratifiedKFold

from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import balanced_accuracy_score  # for Gini-mean
from sklearn.metrics import roc_curve, auc


def model_prediction(model, x, y, n_splits=5, random_state=42):
	skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
	training_score = []
	testing_score = []
	precision = []
	recall = []
	f1_score_ = []
	roc_auc_scores = []
	x = pd.DataFrame(x)
	for train_index, test_index in skf.split(x, y):
		x_train, x_test = x.iloc[train_index], x.iloc[test_index]
		y_train, y_test = y.iloc[train_index], y.iloc[test_index]
		model.fit(x_train, y_train)
		x_train_pred = model.predict(x_train)
		y_test_pred = model.predict(x_test)
		a = accuracy_score(y_train, x_train_pred) * 100
		b = accuracy_score(y_test, y_test_pred) * 100
		c = precision_score(y_test, y_test_pred, average='weighted')
		d = recall_score(y_test, y_test_pred, average='weighted')
		e = f1_score(y_test, y_test_pred, average='weighted')
		# Calculate AUC-ROC score
		roc_auc = roc_auc_score(y_test, model.predict_proba(x_test)[:, 1])
		roc_auc_scores.append(roc_auc)

		training_score.append(a)
		testing_score.append(b)
		precision.append(c)
		recall.append(d)
		f1_score_.append(e)

	print("\n------------------------------------------------------------------------")
	print(f"Mean Accuracy_Score of {model} model on Training Data is:", np.mean(training_score))
	print(f"Mean Accuracy_Score of {model} model on Testing Data is:", np.mean(testing_score))
	print(f"Mean Precision Score of {model} model is:", np.mean(precision))
	print(f"Mean Recall Score of {model} model is:", np.mean(recall))
	print(f"Mean f1 Score of {model} model is:", np.mean(f1_score_))
	print(f"Mean AUC-ROC Score of {model} model is:", np.mean(roc_auc_scores))

	print("\n------------------------------------------------------------------------")
	print(f"Classification Report of {model} model is:")
	y_pred_all = cross_val_predict(model, x, y, cv=skf)
	print(classification_report(y, y_pred_all))

	print("\n------------------------------------------------------------------------")
	print(f"Plotting ROC-AUC curve for {model} model:")
	mean_fpr = np.linspace(0, 1, 100)
	tprs = []
	for train_index, test_index in skf.split(x, y):
		x_train, x_test = x.iloc[train_index], x.iloc[test_index]
		y_train, y_test = y.iloc[train_index], y.iloc[test_index]
		probas_ = model.fit(x_train, y_train).predict_proba(x_test)
		fpr, tpr, thresholds = roc_curve(y_test, probas_[:, 1])
		tprs.append(np.interp(mean_fpr, fpr, tpr))
	mean_tpr = np.mean(tprs, axis=0)
	mean_tpr[-1] = 1.0
	mean_auc = auc(mean_fpr, mean_tpr)
	# plt.plot(mean_fpr, mean_tpr, color='b', label=f'Mean ROC (AUC={mean_auc:.2f}')
	# plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='change', alpha=.8)
	# plt.xlabel('False Positive Rate')
	# plt.ylabel('True Positive Rate')
	# plt.title(f'ROC - AUC Curve for {model} Model')
	# plt.legend(loc='lower right')
	# plt.show()


model_prediction(LogisticRegression(), x_scaled, y1, n_splits=5, random_state=42)
print('0' * 100)
model_prediction(LogisticRegression(), x_scaled, y2, n_splits=5, random_state=42)
print('1' * 100)
model_prediction(LogisticRegression(), x_scaled, y3, n_splits=5, random_state=42)
print('2' * 100)
model_prediction(LogisticRegression(), x_scaled, y4, n_splits=5, random_state=42)
print('3' * 100)
model_prediction(LogisticRegression(), x_scaled, y5, n_splits=5, random_state=42)
print('4' * 100)

model_prediction(LogisticRegression(), x_scaled, y6, n_splits=5, random_state=42)
print('5' * 100)

model_prediction(LogisticRegression(), x_scaled, y7, n_splits=5, random_state=42)
print('6' * 100)


def objective(trial):
	max_depth = trial.suggest_int('max_depth', 3, 10)
	n_estimators = trial.suggest_int('n_estimators', 100, 2000)
	gamma = trial.suggest_float('gamma', 0, 1)
	reg_alpha = trial.suggest_float('reg_alpha', 0, 2)
	reg_lambda = trial.suggest_float('reg_lambda', 0, 2)
	min_child_weight = trial.suggest_int('min_child_weight', 0, 10)
	subsample = trial.suggest_float('subsample', 0, 1)
	colsample_bytree = trial.suggest_float('colsample_bytree', 0, 1)
	learning_rate = trial.suggest_float('learning_rate', 0.01, 1)

	print('Training the model with', x.shape[1], 'features')

	params = {'n_estimators': n_estimators,
			  'learning_rate': learning_rate,
			  'gamma': gamma,
			  'reg_alpha': reg_alpha,
			  'reg_lambda': reg_lambda,
			  'max_depth': max_depth,
			  'min_child_weight': min_child_weight,
			  'subsample': subsample,
			  'colsample_bytree': colsample_bytree,
			  'eval_metric': 'logloss'}  # Using logloss for binary classification

	clf = XGBClassifier(**params,
						booster='gbtree',
						objective='binary:logistic',  # Binary classification objective
						verbosity=0)

	cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
	cv_results = cross_val_score(clf, x, y7, cv=cv, scoring='roc_auc')  # Using roc_auc scoring

	validation_score = np.mean(cv_results)
	return validation_score


study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=25, n_jobs=16)
best_params = study.best_params
print("Best Hyperparameters y7:", best_params)


xgb_best_params_for_y1 = {'max_depth': 5, \
						  'n_estimators': 1627, \
						  'gamma': 0.8952807768735265,
						  'reg_alpha': 1.6314226873472901, \
						  'reg_lambda': 1.7229132141868826, \
						  'min_child_weight': 9,
						  'subsample': 0.9885054042421748, \
						  'colsample_bytree': 0.22439719563481197, \
						  'learning_rate': 0.10650804734533341}
xgb_model_for_y1 = XGBClassifier(**xgb_best_params_for_y1)
result = xgb_model_for_y1.fit(x, y1)
print('7' * 100)
print(result)

# feature importance

feature_importance = xgb_model_for_y1.feature_importances_
feature_importance_df = pd.DataFrame({'Feature': x.columns, 'Importance': feature_importance})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

plt.figure(figsize=(7, 7))
sns.barplot(x='Importance', y='Feature', data=feature_importance_df)
plt.title('Feature Importance')
plt.xlabel('Importance')
plt.ylabel('')
sns.despine(left=True, bottom=True)
plt.show()

xgb_best_params_for_y2 = {'max_depth': 4, 'n_estimators': 623, \
						  'gamma': 0.8519204218831254, \
						  'reg_alpha': 1.2439917683504533, \
						  'reg_lambda': 1.4590567435160746, \
						  'min_child_weight': 8, \
						  'subsample': 0.40710690255500565, \
						  'colsample_bytree': 0.2267807727315173, \
						  'learning_rate': 0.04570427430948454}
xgb_model_for_y2 = XGBClassifier(**xgb_best_params_for_y2)
xgb_model_for_y2.fit( x, y2)
# feature importance
feature_importance = xgb_model_for_y2.feature_importances_
feature_importance_df = pd.DataFrame({"Feature": x.columns, 'Importance': feature_importance})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
plt.figure(figsize=(7, 7))
sns.barplot(x='Importance', y='Feature', data=feature_importance_df)
plt.title('Feature Importance')
plt.xlabel('Importance')
plt.ylabel('')
sns.despine(left=True, bottom=True)
plt.show()
xgb_best_params_for_y3 = {'max_depth': 5, 'n_estimators': 1680, 'gamma': 0.6923689614425462,
						  'reg_alpha': 0.9189470702166882, 'reg_lambda': 1.5117758160539976, 'min_child_weight': 9,
						  'subsample': 0.6940678483755448, 'colsample_bytree': 0.8761358304654752,
						  'learning_rate': 0.011025136150862678}
xgb_model_for_y3 = XGBClassifier(**xgb_best_params_for_y3)
xgb_model_for_y3.fit(x, y3)
# feature importances
feature_importance = xgb_model_for_y3.feature_importances_
feature_importance_df = pd.DataFrame({'Feature': x.columns, 'Importance': feature_importance})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
plt.figure(figsize=(7, 7))
sns.barplot(x='Importance', y='Feature', data=feature_importance_df)
plt.title('Feature Importance')
plt.xlabel('Importance')
plt.ylabel('')
sns.despine(left=True, bottom=True)
plt.show()

xgb_best_params_for_y4 = {'max_depth': 6, 'n_estimators': 401, 'gamma': 0.3226754521662682,
						  'reg_alpha': 0.35015352024357355, 'reg_lambda': 1.455091751574945, 'min_child_weight': 2,
						  'subsample': 0.6613340923578201, 'colsample_bytree': 0.6369472068920922,
						  'learning_rate': 0.02173505504016533}
xgb_model_for_y4 = XGBClassifier(**xgb_best_params_for_y4)
xgb_model_for_y4.fit(x, y4)

# feature importances
feature_importance = xgb_model_for_y4.feature_importances_
feature_importance_df = pd.DataFrame({'Feature': x.columns, 'Importance': feature_importance})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
plt.figure(figsize=(7, 7))
sns.barplot(x='Importance', y='Feature', data=feature_importance_df)
plt.title('Feature Importance')
plt.xlabel('Importance')
plt.ylabel('')
sns.despine(left=True, bottom=True)
plt.show()

xgb_best_params_for_y5 = {'max_depth': 8, 'n_estimators': 1586, 'gamma': 0.32950059377825075,
						  'reg_alpha': 1.9609119815708795, 'reg_lambda': 1.528942899424126, 'min_child_weight': 0,
						  'subsample': 0.2571147836064856, 'colsample_bytree': 0.24989821475746465,
						  'learning_rate': 0.01350991516826753}
xgb_model_for_y5 = XGBClassifier(**xgb_best_params_for_y5)
xgb_model_for_y5.fit(x, y5)
# feature importances
feature_importance = xgb_model_for_y5.feature_importances_
feature_importance_df = pd.DataFrame({'Feature': x.columns, 'Importance': feature_importance})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
plt.figure(figsize=(7, 7))
sns.barplot(x='Importance', y='Feature', data=feature_importance_df)
plt.title('Feature Importance')
plt.xlabel('Importance')
plt.ylabel('')
sns.despine(left=True, bottom=True)
plt.show()

xgb_best_params_for_y6 = {'max_depth': 3, 'n_estimators': 1965, 'gamma': 0.7461420398485773,
						  'reg_alpha': 0.6331839468092292, 'reg_lambda': 1.7474555338548388, 'min_child_weight': 3,
						  'subsample': 0.44572949961178254, 'colsample_bytree': 0.44437417147066066,
						  'learning_rate': 0.013061101850914858}
xgb_model_for_y6 = XGBClassifier(**xgb_best_params_for_y6)
xgb_model_for_y6.fit(x, y6)
# feature importances
feature_importance = xgb_model_for_y6.feature_importances_
feature_importance_df = pd.DataFrame({'Feature': x.columns, 'Importance': feature_importance})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
plt.figure(figsize=(7, 7))
sns.barplot(x='Importance', y='Feature', data=feature_importance_df)
plt.title('Feature Importance')
plt.xlabel('Importance')
plt.ylabel('')
sns.despine(left=True, bottom=True)
plt.show()
xgb_best_params_for_y7 = {'max_depth': 4, 'n_estimators': 663, 'gamma': 0.6429564571848232,
						  'reg_alpha': 0.3267006339507057, 'reg_lambda': 0.04658361960102192, 'min_child_weight': 6,
						  'subsample': 0.9939674566310442, 'colsample_bytree': 0.1435958193323451,
						  'learning_rate': 0.24960789830790053}
xgb_model_for_y7 = XGBClassifier(**xgb_best_params_for_y7)
xgb_model_for_y7.fit(x, y7)

# feature importances
feature_importance = xgb_model_for_y7.feature_importances_
feature_importance_df = pd.DataFrame({'Feature': x.columns, 'Importance': feature_importance})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
plt.figure(figsize=(7, 7))
sns.barplot(x='Importance', y='Feature', data=feature_importance_df)
plt.title('Feature Importance')
plt.xlabel('Importance')
plt.ylabel('')
sns.despine(left=True, bottom=True)
plt.show()


model_prediction(XGBClassifier(**xgb_best_params_for_y1), x, y1, n_splits=5, random_state=42)
model_prediction(XGBClassifier(**xgb_best_params_for_y2), x, y2, n_splits=5, random_state=42)
model_prediction(XGBClassifier(**xgb_best_params_for_y3), x, y3, n_splits=5, random_state=42)
model_prediction(XGBClassifier(**xgb_best_params_for_y4), x, y4, n_splits=5, random_state=42)
model_prediction(XGBClassifier(**xgb_best_params_for_y5), x, y5, n_splits=5, random_state=42)
model_prediction(XGBClassifier(**xgb_best_params_for_y6), x, y6, n_splits=5, random_state=42)
model_prediction(XGBClassifier(**xgb_best_params_for_y7), x, y7, n_splits=5, random_state=42)

'''def objective(trial):
    learning_rate = trial.suggest_float('learning_rate', 0.01, 0.1)
    iterations = trial.suggest_int('iterations', 100, 5000)
    depth = trial.suggest_int('depth', 3, 10)
    colsample_bylevel = trial.suggest_float('colsample_bylevel', 0.5, 1.0)
    l2_leaf_reg = trial.suggest_float('l2_leaf_reg', 0.0, 2.0)
    border_count = trial.suggest_int('border_count', 32, 255)

    print('Training the model with', x.shape[1], 'features')

    params = {
        'learning_rate': learning_rate,
        'iterations': iterations,
        'depth': depth,
        'colsample_bylevel': colsample_bylevel,
        'l2_leaf_reg': l2_leaf_reg,
        'border_count': border_count,
    }

    clf = CatBoostClassifier(**params, eval_metric='AUC', task_type="CPU", verbose=300, early_stopping_rounds=50)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_results = cross_validate(clf, x, y7, cv=cv, scoring='roc_auc')

    validation_score = np.mean(cv_results['test_score'])

    return validation_score'''
# Set up Optuna study
# study = optuna.create_study(direction='maximize')
# study.optimize(objective, n_trials=20)

# Get the best hyperparameters
# best_params = study.best_params
# print("Best Hyperparameters for y7:", best_params)
catboost_best_params_for_y1 = {'learning_rate': 0.0293790855184329, 'iterations': 1481, 'depth': 3,
							   'colsample_bylevel': 0.5711312763952309, 'l2_leaf_reg': 1.5994222344635594,
							   'border_count': 208}
cb_model_for_y1 = CatBoostClassifier(**catboost_best_params_for_y1)
cb_model_for_y1.fit(x, y1)
# feature importances
feature_importance = cb_model_for_y1.feature_importances_
feature_importance_df = pd.DataFrame({'Feature': x.columns, 'Importance': feature_importance})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
plt.figure(figsize=(7, 7))
sns.barplot(x='Importance', y='Feature', data=feature_importance_df)
plt.title('Feature Importance')
plt.xlabel('Importance')
plt.ylabel('')
sns.despine(left=True, bottom=True)
plt.show()

catboost_best_params_for_y2 = {'learning_rate': 0.03857054239913968, 'iterations': 206, 'depth': 10,
							   'colsample_bylevel': 0.5393784309306074, 'l2_leaf_reg': 1.3906029897172827,
							   'border_count': 231}
cb_model_for_y2 = CatBoostClassifier(**catboost_best_params_for_y2)
cb_model_for_y2.fit(x, y2)

feature_importance=cb_model_for_y2.feature_importances_
feature_importance_df=pd.DataFrame({'Feature':x.columns,
									'Importance':feature_importance})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
plt.figure(figsize=(7, 7))
sns.barplot(x='Importance', y='Feature', data=feature_importance_df)
plt.title('Feature Importance')
plt.xlabel('Importance')
plt.ylabel('')
sns.despine(left=True, bottom=True)
plt.show()



catboost_best_params_for_y3 = {'learning_rate': 0.05819735650442166, 'iterations': 165, 'depth': 6,
							   'colsample_bylevel': 0.960964834849099, 'l2_leaf_reg': 0.6700019633321236,
							   'border_count': 189}

cb_model_for_y3 = CatBoostClassifier(**catboost_best_params_for_y3)
cb_model_for_y3.fit(x, y3)

# feature importances
feature_importance = cb_model_for_y3.feature_importances_
feature_importance_df = pd.DataFrame({'Feature': x.columns, 'Importance': feature_importance})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
plt.figure(figsize=(7, 7))
sns.barplot(x='Importance', y='Feature', data=feature_importance_df)
plt.title('Feature Importance')
plt.xlabel('Importance')
plt.ylabel('')
sns.despine(left=True, bottom=True)
plt.show()

catboost_best_params_for_y4 = {'learning_rate': 0.032940739491683975, 'iterations': 407, 'depth': 8,
							   'colsample_bylevel': 0.6333926050478358, 'l2_leaf_reg': 1.1045970003458674,
							   'border_count': 130}

cb_model_for_y4 = CatBoostClassifier(**catboost_best_params_for_y4)
cb_model_for_y4.fit(x, y4)
# feature importances
feature_importance = cb_model_for_y4.feature_importances_
feature_importance_df = pd.DataFrame({'Feature': x.columns, 'Importance': feature_importance})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
plt.figure(figsize=(7, 7))
sns.barplot(x='Importance', y='Feature', data=feature_importance_df)
plt.title('Feature Importance')
plt.xlabel('Importance')
plt.ylabel('')
sns.despine(left=True, bottom=True)
plt.show()

catboost_best_params_for_y5 = {'learning_rate': 0.07936470052548467, 'iterations': 135, 'depth': 8,
							   'colsample_bylevel': 0.8032026164713476, 'l2_leaf_reg': 0.7783270167485883,
							   'border_count': 247}
cb_model_for_y5 = CatBoostClassifier(**catboost_best_params_for_y5)
cb_model_for_y5.fit(x, y5)
# feature importances
feature_importance = cb_model_for_y5.feature_importances_
feature_importance_df = pd.DataFrame({'Feature': x.columns, 'Importance': feature_importance})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
plt.figure(figsize=(7, 7))
sns.barplot(x='Importance', y='Feature', data=feature_importance_df)
plt.title('Feature Importance')
plt.xlabel('Importance')
plt.ylabel('')
sns.despine(left=True, bottom=True)
plt.show()

catboost_best_params_for_y6 = {'learning_rate': 0.010315180165549727, 'iterations': 1156, 'depth': 9,
							   'colsample_bylevel': 0.6160311869329463, 'l2_leaf_reg': 1.1668408958633865,
							   'border_count': 179}
cb_model_for_y6 = CatBoostClassifier(**catboost_best_params_for_y6)
cb_model_for_y6.fit(x, y6)

# feature importances
feature_importance = cb_model_for_y6.feature_importances_
feature_importance_df = pd.DataFrame({'Feature': x.columns, 'Importance': feature_importance})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
plt.figure(figsize=(7, 7))
sns.barplot(x='Importance', y='Feature', data=feature_importance_df)
plt.title('Feature Importance')
plt.xlabel('Importance')
plt.ylabel('')
sns.despine(left=True, bottom=True)
plt.show()

catboost_best_params_for_y7 = {'learning_rate': 0.025827131598767934, 'iterations': 848, 'depth': 3,
							   'colsample_bylevel': 0.74491431963177, 'l2_leaf_reg': 0.12688953370416511,
							   'border_count': 189}
cb_model_for_y7 = CatBoostClassifier(**catboost_best_params_for_y7)
cb_model_for_y7.fit(x, y7)
# feature importances
feature_importance = cb_model_for_y7.feature_importances_
feature_importance_df = pd.DataFrame({'Feature': x.columns, 'Importance': feature_importance})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
plt.figure(figsize=(7, 7))
sns.barplot(x='Importance', y='Feature', data=feature_importance_df)
plt.title('Feature Importance')
plt.xlabel('Importance')
plt.ylabel('')
sns.despine(left=True, bottom=True)
plt.show()


'''def objective(trial):
    params = {
        'objective': 'binary',
        'metric': 'auc',
        'verbosity': -1,
        'boosting_type': 'gbdt',
        'lambda_l1': 0.0,
        'lambda_l2': 0.0,
        'num_leaves': trial.suggest_int('num_leaves', 10, 200),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.1, 1.0),
        'subsample': trial.suggest_float('subsample', 0.1, 1.0),
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 10),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1)
    }

    print('Training the model with', x.shape[1], 'features')

    lgb_classifier = LGBMClassifier(**params)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_results = cross_val_score(lgb_classifier, x, y7, cv=cv, scoring='roc_auc')

    validation_score = np.mean(cv_results)

    return validation_score'''

# Set up Optuna study
# study = optuna.create_study(direction='maximize')
# study.optimize(objective, n_trials=100)

# Get the best hyperparameters
# best_params = study.best_params
# print("Best Hyperparameters for y7:", best_params)

lgbm_best_params_for_y1 = {'num_leaves': 10, 'colsample_bytree': 0.4938794627384321,
						   'subsample': 0.8742780567743516,
						   'bagging_freq': 8, 'min_child_samples': 71,
						   'learning_rate': 0.07912118780949727}
lgbm_model_for_y1 = LGBMClassifier(**lgbm_best_params_for_y1)
lgbm_model_for_y1.fit( x, y1)

# feature importances
feature_importance = lgbm_model_for_y1.feature_importances_
feature_importance_df = pd.DataFrame({'Feature': x.columns, 'Importance': feature_importance})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
plt.figure(figsize=(7, 7))
sns.barplot(x='Importance', y='Feature', data=feature_importance_df)
plt.title('Feature Importance')
plt.xlabel('Importance')
plt.ylabel('')
sns.despine(left=True, bottom=True)
plt.show()

lgbm_best_params_for_y2 = {'num_leaves': 21, 'colsample_bytree': 0.4151347339206033, 'subsample': 0.8370319196132707, 'bagging_freq': 6, 'min_child_samples': 29, 'learning_rate': 0.04847995943398712}

lgbm_model_for_y2 = LGBMClassifier(**lgbm_best_params_for_y2)
lgbm_model_for_y2.fit( x, y2)

# feature importances
feature_importance = lgbm_model_for_y2.feature_importances_
feature_importance_df = pd.DataFrame({'Feature': x.columns, 'Importance': feature_importance})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
plt.figure(figsize=(7, 7))
sns.barplot(x='Importance', y='Feature', data=feature_importance_df)
plt.title('Feature Importance')
plt.xlabel('Importance')
plt.ylabel('')
sns.despine(left=True, bottom=True)
plt.show()

lgbm_best_params_for_y3 = {'num_leaves': 100, 'colsample_bytree': 0.5087492462560602, 'subsample': 0.4411107602895591, 'bagging_freq': 1, 'min_child_samples': 62, 'learning_rate': 0.038454266176433355}
lgbm_model_for_y3 = LGBMClassifier(**lgbm_best_params_for_y3)
lgbm_model_for_y3.fit( x, y3)

# feature importances
feature_importance = lgbm_model_for_y3.feature_importances_
feature_importance_df = pd.DataFrame({'Feature': x.columns, 'Importance': feature_importance})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
plt.figure(figsize=(7, 7))
sns.barplot(x='Importance', y='Feature', data=feature_importance_df)
plt.title('Feature Importance')
plt.xlabel('Importance')
plt.ylabel('')
sns.despine(left=True, bottom=True)
plt.show()

lgbm_best_params_for_y4 = {'num_leaves': 131, 'colsample_bytree': 0.5697305625366826, 'subsample': 0.8109563189916542, 'bagging_freq': 4, 'min_child_samples': 19, 'learning_rate': 0.05213996016102166}
lgbm_model_for_y4 = LGBMClassifier(**lgbm_best_params_for_y4)
lgbm_model_for_y4.fit( x, y4)
# feature importances
feature_importance = lgbm_model_for_y4.feature_importances_
feature_importance_df = pd.DataFrame({'Feature': x.columns, 'Importance': feature_importance})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
plt.figure(figsize=(7, 7))
sns.barplot(x='Importance', y='Feature', data=feature_importance_df)
plt.title('Feature Importance')
plt.xlabel('Importance')
plt.ylabel('')
sns.despine(left=True, bottom=True)
plt.show()



lgbm_best_params_for_y5 = {'num_leaves': 91, 'colsample_bytree': 0.1639450170074021, 'subsample': 0.5341894798705625, 'bagging_freq': 4, 'min_child_samples': 47, 'learning_rate': 0.018293880009565062}


lgbm_model_for_y5 = LGBMClassifier(**lgbm_best_params_for_y5)
lgbm_model_for_y5.fit( x, y5)

# feature importances
feature_importance = lgbm_model_for_y5.feature_importances_
feature_importance_df = pd.DataFrame({'Feature': x.columns, 'Importance': feature_importance})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
plt.figure(figsize=(7, 7))
sns.barplot(x='Importance', y='Feature', data=feature_importance_df)
plt.title('Feature Importance')
plt.xlabel('Importance')
plt.ylabel('')
sns.despine(left=True, bottom=True)
plt.show()


lgbm_best_params_for_y6 =  {'num_leaves': 22, 'colsample_bytree': 0.667438994216906, 'subsample': 0.8090648564857341, 'bagging_freq': 1, 'min_child_samples': 93, 'learning_rate': 0.06318737564630748}
lgbm_model_for_y6 = LGBMClassifier(**lgbm_best_params_for_y6)
lgbm_model_for_y6.fit( x, y6)
# feature importances
feature_importance = lgbm_model_for_y6.feature_importances_
feature_importance_df = pd.DataFrame({'Feature': x.columns, 'Importance': feature_importance})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
plt.figure(figsize=(7, 7))
sns.barplot(x='Importance', y='Feature', data=feature_importance_df)
plt.title('Feature Importance')
plt.xlabel('Importance')
plt.ylabel('')
sns.despine(left=True, bottom=True)
plt.show()

lgbm_best_params_for_y7 = {'num_leaves': 16, 'colsample_bytree': 0.5430173377906267, 'subsample': 0.9060940434087468, 'bagging_freq': 2, 'min_child_samples': 46, 'learning_rate': 0.04345358731901847}
lgbm_model_for_y7 = LGBMClassifier(**lgbm_best_params_for_y7)
lgbm_model_for_y7.fit( x, y7)
# feature importances
feature_importance = lgbm_model_for_y7.feature_importances_
feature_importance_df = pd.DataFrame({'Feature': x.columns, 'Importance': feature_importance})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
plt.figure(figsize=(7, 7))
sns.barplot(x='Importance', y='Feature', data=feature_importance_df)
plt.title('Feature Importance')
plt.xlabel('Importance')
plt.ylabel('')
sns.despine(left=True, bottom=True)
plt.show()

ensemble_model_for_y1 = VotingClassifier(
    estimators=[
        ('xgb', xgb_model_for_y1),
        ('catboost',cb_model_for_y1),
        ('LGBM', lgbm_model_for_y1)
    ],
    voting='soft',
    flatten_transform=True,
	n_jobs=16
)

# Use AUC-ROC as the scoring parameter
auc_roc_scores = cross_val_score(ensemble_model_for_y1, x, y1, cv=5, scoring='roc_auc')

print("AUC-ROC scores for each fold:", auc_roc_scores)
print("Average AUC-ROC:", auc_roc_scores.mean())


ensemble_model_for_y1.fit(x,y1)
Pastry = ensemble_model_for_y1.predict_proba(test_df)
Pastry = Pastry[:,1]
Pastry

ensemble_model_for_y2 = VotingClassifier(
    estimators=[
        ('xgb', xgb_model_for_y2),
        ('catboost',cb_model_for_y2),
        ('LGBM', lgbm_model_for_y2)
    ],
    voting='soft',
    flatten_transform=True,
	n_jobs=16
)

# Use AUC-ROC as the scoring parameter
auc_roc_scores = cross_val_score(ensemble_model_for_y2, x, y2, cv=5, scoring='roc_auc')

print("AUC-ROC scores for each fold:", auc_roc_scores)
print("Average AUC-ROC:", auc_roc_scores.mean())

ensemble_model_for_y2.fit(x,y2)
Z_Scratch = ensemble_model_for_y2.predict_proba(test_df)
Z_Scratch = Z_Scratch[:,1]

ensemble_model_for_y3 = VotingClassifier(
    estimators=[
        ('xgb', xgb_model_for_y3),
        ('catboost',cb_model_for_y3),
        ('LGBM', lgbm_model_for_y3)
    ],
    voting='soft',
    flatten_transform=True
)

# Use AUC-ROC as the scoring parameter
auc_roc_scores = cross_val_score(ensemble_model_for_y3, x, y3, cv=5, scoring='roc_auc')

print("AUC-ROC scores for each fold:", auc_roc_scores)
print("Average AUC-ROC:", auc_roc_scores.mean())
ensemble_model_for_y3.fit(x,y3)

K_Scatch = ensemble_model_for_y3.predict_proba(test_df)
K_Scatch = K_Scatch[:,1]
K_Scatch

ensemble_model_for_y4 = VotingClassifier(
    estimators=[
        ('xgb', xgb_model_for_y4),
        ('catboost',cb_model_for_y4),
        ('LGBM', lgbm_model_for_y4)
    ],
    voting='soft',
    flatten_transform=True,
	n_jobs=16
)

# Use AUC-ROC as the scoring parameter
auc_roc_scores = cross_val_score(ensemble_model_for_y4, x, y4, cv=5, scoring='roc_auc')

print("AUC-ROC scores for each fold:", auc_roc_scores)
print("Average AUC-ROC:", auc_roc_scores.mean())

ensemble_model_for_y4.fit(x,y4)
Stains = ensemble_model_for_y4.predict_proba(test_df)
Stains = Stains[:,1]
Stains

ensemble_model_for_y5 = VotingClassifier(
    estimators=[
        ('xgb', xgb_model_for_y5),
        ('catboost',cb_model_for_y5),
        ('LGBM', lgbm_model_for_y5)
    ],
    voting='soft',
    flatten_transform=True
)

# Use AUC-ROC as the scoring parameter
auc_roc_scores = cross_val_score(ensemble_model_for_y5, x, y5, cv=5, scoring='roc_auc')

print("AUC-ROC scores for each fold:", auc_roc_scores)
print("Average AUC-ROC:", auc_roc_scores.mean())
ensemble_model_for_y5.fit(x,y5)
Dirtiness = ensemble_model_for_y5.predict_proba(test_df)
Dirtiness = Dirtiness[:,1]
Dirtiness

ensemble_model_for_y6 = VotingClassifier(
    estimators=[
        ('xgb', xgb_model_for_y6),
        ('catboost',cb_model_for_y6),
        ('LGBM', lgbm_model_for_y6)
    ],
    voting='soft',
    flatten_transform=True
)

# Use AUC-ROC as the scoring parameter
auc_roc_scores = cross_val_score(ensemble_model_for_y6, x, y6, cv=5, scoring='roc_auc')

print("AUC-ROC scores for each fold:", auc_roc_scores)
print("Average AUC-ROC:", auc_roc_scores.mean())
ensemble_model_for_y6.fit(x,y6)
Bumps = ensemble_model_for_y6.predict_proba(test_df)
Bumps = Bumps[:,1]
Bumps
ensemble_model_for_y7 = VotingClassifier(
    estimators=[
        ('xgb', xgb_model_for_y7),
        ('catboost',cb_model_for_y7),
        ('LGBM', lgbm_model_for_y7)
    ],
    voting='soft',
    flatten_transform=True
)

# Use AUC-ROC as the scoring parameter
auc_roc_scores = cross_val_score(ensemble_model_for_y7, x, y7, cv=5, scoring='roc_auc')

print("AUC-ROC scores for each fold:", auc_roc_scores)
print("Average AUC-ROC:", auc_roc_scores.mean())

ensemble_model_for_y7.fit(x,y7)
Other_Faults = ensemble_model_for_y7.predict_proba(test_df)
Other_Faults = Other_Faults[:,1]
Other_Faults

sample_submission = pd.read_csv('/kaggle/input/playground-series-s4e3/sample_submission.csv')
sample_submission
sample_submission['Pastry'] = Pastry
sample_submission['Z_Scratch'] = Z_Scratch
sample_submission['K_Scatch'] = K_Scatch
sample_submission['Stains'] = Stains
sample_submission['Dirtiness'] = Dirtiness
sample_submission['Bumps'] = Bumps
sample_submission['Other_Faults'] = Other_Faults
sample_submission

sample_submission.to_csv("submission.csv", index=False)





