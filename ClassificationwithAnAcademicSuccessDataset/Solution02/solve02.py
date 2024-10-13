# -*- coding: utf-8 -*-
# @Time    : 2024/6/10 下午4:28
# @Author  : nanji
# @Site    : https://www.kaggle.com/code/abdmental01/lgbm-optimization-optuna
# @File    : solve02.py
# @Software: PyCharm 
# @Comment : 10.11

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math
import numpy as np
from colorama import Fore, Style, init
import optuna

import warnings

warnings.filterwarnings('ignore')
# Model and utiliz
from sklearn.model_selection import RepeatedStratifiedKFold, \
    cross_val_score, \
    train_test_split
import lightgbm as lgb
from lightgbm import LGBMClassifier
from sklearn.preprocessing import *
from sklearn.metrics import *

import warnings

warnings.filterwarnings('ignore')

# Set the option to display all columns
pd.set_option('display.max_columns', None)
# Intlize Colors
HEAD = '#FFFF00'
TEXT = '#2DA8D8'


def print_unique_header(heading, heading_color=HEAD, text_color=TEXT):
    def color_text(text, hex_color):
        # Convert hex color to RGB
        rgb = tuple(int(hex_color[i:i + 1], 16) for i in (1, 3, 5))
        # Apply ANSI escape code for the given RGB color
        return f"\033[38;2;{rgb[0]};{rgb[1]};{rgb[2]}m{text}\033[0m"

    bright = "\033[1m"
    reset = "\033[0m"
    total_width = len(heading) + 20
    left_space = (total_width - len(heading)) // 2
    right_space = total_width - len(heading) - left_space
    print("\n" + color_text("╭" + "─" * total_width + "╮", heading_color))
    print(color_text(f"│{' ' * left_space}{'▲'}{' ' * right_space}",
                     heading_color) + reset)
    print(color_text(f"│{' ' * left_space}", heading_color) +
          color_text(heading, text_color) + color_text(
        f"{' ' * right_space}│", heading_color) + reset)
    print(color_text(f"│{' ' * left_space}{'▼'}{' ' * right_space}", heading_color)
          + reset)
    print(color_text("╰" + "─" * total_width + "╯", heading_color))


def print_boxed_zigzag_heading(heading, heading_color=HEAD, text_color=TEXT):
    def color_text(text, hex_color):
        # Convert hex color to RGB
        rgb = tuple(int(hex_color[i:i + 2], 16) for i in (1, 3, 5))
        # Apply ANSI escape code for the given RGB color
        return f"\033[38;2;{rgb[0]};{rgb[1]};{rgb[2]}m{text}\033[0m"

    bright = "\033[1m"
    reset = "\033[0m"
    heading_color_code = color_text("╭" + "─" * (len(heading) + 20) + "╮", heading_color)
    print("\n" + heading_color_code)
    words = heading.split()
    for i, word in enumerate(words):
        if i == len(words) - 1:
            print(f"{color_text(f'│ {word} │', text_color)}{reset}")
        else:
            print(f"{color_text(f'│ {word}', text_color)}{reset}", end=" ")
    print(color_text("╰" + "─" * (len(heading) + 20) + "╯", heading_color))


def prinT(text, hex_color=TEXT):
    # Convert hex color to RGB
    rgb = tuple(int(hex_color[i:i + 2], 16) for i in (1, 3, 5))
    # Apply ANSI escape code for the given RGB color
    colored_text = f"\033[38;2;{rgb[0]};{rgb[1]};{rgb[2]}m{text}\033[0m"
    print(colored_text)


print_unique_header("Setup Intilized")

# Load Data || Feature Engineering


# Load Submission Data
d_s = pd.read_csv('../input/playground-series-s4e6/sample_submission.csv')
# Load test Data
te_d = pd.read_csv('../input/playground-series-s4e6/test.csv')
# Train Data
tr_d = pd.read_csv('../input/playground-series-s4e6/train.csv')

# Original Data
O_D = pd.read_csv("../input/playgrounds4e06originaldata/original.csv")
# Dropping Id from  Train
tr_d.drop(columns=['id'], inplace=True)
te_d.drop(columns=['id'], inplace=True)
O_D.drop(columns=['id'], inplace=True)

# Concat
tr_d = pd.concat(objs=[tr_d, O_D])

# Drop Irreleveant
tr_d.drop(labels='Daytime/evening attendance\t', axis=1, inplace=True)
# Fill Null Values
most_frequent = tr_d['Daytime/evening attendance'].mode()[0]
tr_d['Daytime/evening attendance'].fillna(most_frequent, inplace=True)


# Overview Fucntion
def print_boxed_blue_heading(heading):
    gradient = [Fore.BLUE, Fore.CYAN, Fore.GREEN, Fore.YELLOW, Fore.RED, Fore.MAGENTA]
    print("\n" + "=" * (len(heading) + 4))
    words = heading.split()
    for i, word in enumerate(words):
        if i == len(words) - 1:
            print(f"| {gradient[len(word) % len(gradient)] + word + Style.RESET_ALL} |")
        else:
            print(f"| {gradient[len(word) % len(gradient)] + word + Style.RESET_ALL}", end=" ")
    print("=" * (len(heading) + 4))


print_unique_header('Data Loaded')
# Unique value counts for
U_C = tr_d.nunique()
# Threshold to distinguish continuous and categorical
threshold = 12
Num_C = U_C[U_C > threshold].index.to_list()
Cat_C = U_C[U_C <= threshold].index.to_list()
if 'Target' in Cat_C:
    Cat_C.remove('Target')

print('Num_C:')
print(Num_C)
print('Cat_C:')
print(Cat_C)
# Scale Data Train and Test
S = StandardScaler()
tr_d[Num_C] = S.fit_transform(tr_d[Num_C])
te_d[Num_C] = S.transform(te_d[Num_C])
print_unique_header('Data Scaled')

# Encode Target
Le = LabelEncoder()
tr_d['Target'] = Le.fit_transform(tr_d['Target'])
print_unique_header('Target Encoded')

# # # =================================================================================================================
# # #                         X < y
# # #==================================================================================================================
X_T = tr_d.drop('Target', axis=1)
y_T = tr_d['Target']

# # # =================================================================================================================
# # #                         Train < Test Split
# # #==================================================================================================================
X_TR, X_TE, Y_TR, Y_TE = train_test_split(X_T, y_T, test_size=0.1, random_state=42)
# # # =================================================================================================================
# # #                         Shapes <
# # #==================================================================================================================
print_unique_header(f"Training set shape - X: {X_TR.shape}, y: {Y_TR.shape}")
print_unique_header(f"Testing set shape - X: {X_TE.shape}, y: {Y_TE.shape}")


# Define the objective function
def objective(trial):
    # Define parameters to be optimized
    params = {
        'num_leaves': trial.suggest_int('num_leaves', 100, 1000),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 1.0, log=True),
        'n_estimators': trial.suggest_int('n_estimators', 300, 1200),
        'subsample_for_bin': trial.suggest_int('subsample_for_bin', 20000, 300000),
        'min_child_samples': trial.suggest_int('min_child_samples', 20, 500),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-9, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-9, 10.0, log=True),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.4, 1.0),
        'subsample': trial.suggest_float('subsample', 0.25, 1.0),
        'max_depth': trial.suggest_int('max_depth', 1, 50),
        'num_class': 3
    }
    # Train the model
    model = LGBMClassifier(**params, objective='multiclass',
                           random_state=0, device='cpu', n_jobs=8,
                           verbosity=-1)
    model.fit(X_TR, Y_TR)
    # Evaluate the model on the training data
    y_pred = model.predict(X_TE)
    acc = accuracy_score(Y_TE, y_pred)
    return acc


study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=200)
# Get the best parameters and best accuracy score
best_params = study.best_params
best_acc = study.best_value
print('best accuracy score.', best_acc)
print('best parameters.', best_params)


def Cross_Validate_R(classifier, params, SCORING, N_CV):
    # Train model
    BASE = classifier(**params, objective='multiclass', random_state=0,
                      device='cpu', n_jobs=16, verbosity=-1)
    CV = cross_val_score(BASE,
                         X_T,
                         y_T,
                         scoring=SCORING,
                         cv=N_CV,
                         n_jobs=8
                         )
    # Print results
    print_boxed_blue_heading(f"The Accuracy of {classifier.__name__} Classifier is: {CV.mean()}")
    return BASE


# Parameters tuned with Optuna [LB : Score: 0.83581]
# L_params =  {'num_leaves': 175, 'learning_rate': 0.023538121810223783, 'n_estimators': 1014,
#              'subsample_for_bin': 82512, 'min_child_samples': 412, 'reg_alpha': 3.564652653541536e-06,
#              'reg_lambda': 8.292918665170978e-06, 'colsample_bytree': 0.4998959438718756,
#              'subsample': 0.746323514536357, 'max_depth': 4, 'num_class': 3}
L_params = {'num_leaves': 974, 'learning_rate': 0.029398425765109985,  #
            'n_estimators': 698,  #
            'subsample_for_bin': 170570, 'min_child_samples': 98,  #
            'reg_alpha': 0.029613659564849536,  #
            'reg_lambda': 0.11430080990760436,  #
            'colsample_bytree': 0.6433250504280341,  #
            'subsample': 0.9008773629742881,  #
            'max_depth': 11, 'num_class': 3}  #

# Adjust scoring and number of cross-validation folds
SCORING = 'accuracy'
N_CV = 20

# Perform cross-validation
BASE = Cross_Validate_R(LGBMClassifier, L_params, SCORING, N_CV)
# Fit Again
BASE.fit(X_T, y_T)
# Test Pred
L_Pred = BASE.predict(te_d)
# Marking Submission File
d_s['Target'] = Le.inverse_transform(L_Pred)
d_s.to_csv('L_Submission3.csv', index=False)
print_unique_header('Submission File Saved')

