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