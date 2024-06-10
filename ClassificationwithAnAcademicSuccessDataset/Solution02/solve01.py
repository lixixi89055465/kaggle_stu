# -*- coding: utf-8 -*-
# @Time    : 2024/6/9 下午9:25
# @Author  : nanji
# @Site    : https://www.kaggle.com/code/abdmental01/ensemble-solution-top-3
# @File    : solve01.py
# @Software: PyCharm 
# @Comment :
# Import
import pandas as pd
import numpy as np
from scipy.stats import mode
from colorama import Fore, Style, init;


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


SUB_F = pd.read_csv(f"../input/playground-series-s4e6/train.csv")
# Blending my 4 Different Works


Ravi = pd.read_csv(f"/kaggle/input/playgrounds4e06-eda-baseline/Submission_V6.csv")['Target']
Satya = pd.read_csv(f"/kaggle/input/eda-droput-graduate-analysis/submission.csv")['Target']
Gaurav = pd.read_csv(f'/kaggle/input/pss4e6-flaml-roc-auc-ovo/roc_auc_ovo.csv')['Target']
Robert = pd.read_csv(f'/kaggle/input/automl-2nd-place-plus-inspect-public-lb-subs/submission.csv')['Target']
Abdullah = pd.read_csv(f"/kaggle/input/submission-abdullah/5_Blend.csv")['Target']

Pred = pd.concat([Ravi, Satya, Gaurav, Robert, Abdullah])

