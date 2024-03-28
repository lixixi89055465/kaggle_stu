# -*- coding: utf-8 -*-
# @Time : 2024/3/28 11:34
# @Author : nanji
# @Site : https://zhuanlan.zhihu.com/p/595995679
# @File : test_optuna_pytorch.py
# @Software: PyCharm 
# @Comment : 
import os

import optuna
from optuna.trial import TrialState
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torch.functional as F
from torchvision import datasets
from torchvision import transforms


DEVICE = torch.device('cpu')
BATCHSIZE = 128
CLASSES = 10
DIR = os.getcwd()
EPOCHS = 10
N_TRAIN_EXAMPLES = BATCHSIZE * 38
N_VALID_EXAMPLES = BATCHSIZE * 10


def define_model(trial):
	# We optimize the number of layers, hidden units and dropout ratio in each layer.
	n_layers = trial.suggest_int('n_layers', 1, 3)
	layers = []
	in_features = 28 * 28
	for i in range(n_layers):
		out_features = trial.suggest_int('n_unit_{}'.format(i), 16, 64)
		layers.append(nn.Linear(in_features, out_features))
		layers.append(nn.ReLU())
		p = trial.suggest_float('dropout_{}'.format(i), 0.1, 0.5)
		layers.append(nn.Dropout(p))
		in_features = out_features
	layers.append(nn.Linear(in_features, CLASSES))
	return nn.Sequential(*layers)


def get_mnist():
	# Load FashionMNIST dataset.
	train_loader = torch.utils.data.DataLoader(
		datasets.FashionMNIST(DIR, train=True, download=True, transform=transforms.ToTensor()),
		batch_size=BATCHSIZE,
		shuffle=True,
	)
	valid_loader = torch.utils.data.DataLoader(
		datasets.FashionMNIST(DIR, train=False, transform=transforms.ToTensor()),
		batch_size=BATCHSIZE,
		shuffle=True,
	)
	return train_loader, valid_loader


def objective(trial):
	model = define_model(trial).to(DEVICE)
	optimizer_name = trial.suggest_categorical('optimizer', ["Adam", "RMSprop", "SGD"])
	lr = trial.suggest_float('lr', 1e-5, 1e-1, log=True)
	optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr)
	train_loader, valid_loader = get_mnist()
	for epoch in range(EPOCHS):
		model.train()
		for batch_idx, (data, target) in enumerate(train_loader):
			# Limiting training data for faster epochs.
			if batch_idx*BATCHSIZE>=N_TRAIN_EXAMPLES:
				break
			data,target=data.view(data.size(0),-1).to(DEVICE),target.to(DEVICE)
			optimizer.zero_grad()
			output=model(data)
			loss=F.nll_loss(output,target)
			loss.backward()
			optimizer.step()


if __name__ == '__main__':
	storage_name = "sqlite:///optuna.db"
	study = optuna.create_study(
		pruner=optuna.pruners.MedianPruner(n_warmup_steps=3), direction="maximize",
		study_name="fashion_mnist_torch", storage=storage_name, load_if_exists=True
	)
	study.optimize(objective, n_trials=20, timeout=1200)
