# -*- coding: utf-8 -*-
# @Time : 2024/3/27 23:59
# @Author : nanji
# @Site : https://zhuanlan.zhihu.com/p/595995679
# @File : testoptuna1.py
# @Software: PyCharm 
# @Comment : 
import os

import optuna
from optuna.trial import TrialState
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from torchvision import datasets
from torchvision import transforms

DEVICE = torch.device("cpu")
BATCHSIZE = 128
CLASSES = 10
DIR = os.getcwd()
EPOCHS = 10
N_TRAIN_EXAMPLES = BATCHSIZE * 30
N_VALID_EXAMPLES = BATCHSIZE * 10


def define_model(trial):
	# We optimize the number of layers, hidden units and dropout ratio in each layer.
	n_layers = trial.suggest_int("n_layers", 1, 3)
	layers = []

	in_features = 28 * 28
	for i in range(n_layers):
		out_features = trial.suggest_int("n_units_{}".format(i), 16, 64)
		layers.append(nn.Linear(in_features, out_features))
		layers.append(nn.ReLU())
		p = trial.suggest_float("dropout_{}".format(i), 0.1, 0.5)
		layers.append(nn.Dropout(p))

		in_features = out_features
	layers.append(nn.Linear(in_features, CLASSES))
	layers.append(nn.LogSoftmax(dim=1))

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
	# Generate the model.
	model = define_model(trial).to(DEVICE)

	# Generate the optimizers.
	optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "SGD"])
	lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
	optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr)

	# Get the FashionMNIST dataset.
	train_loader, valid_loader = get_mnist()

	# Training of the model.
	for epoch in range(EPOCHS):
		model.train()
		for batch_idx, (data, target) in enumerate(train_loader):
			# Limiting training data for faster epochs.
			if batch_idx * BATCHSIZE >= N_TRAIN_EXAMPLES:
				break

			data, target = data.view(data.size(0), -1).to(DEVICE), target.to(DEVICE)

			optimizer.zero_grad()
			output = model(data)
			loss = F.nll_loss(output, target)
			loss.backward()
			optimizer.step()

		# Validation of the model.
		model.eval()
		correct = 0
		with torch.no_grad():
			for batch_idx, (data, target) in enumerate(valid_loader):
				# Limiting validation data.
				if batch_idx * BATCHSIZE >= N_VALID_EXAMPLES:
					break
				data, target = data.view(data.size(0), -1).to(DEVICE), target.to(DEVICE)
				output = model(data)
				# Get the index of the max log-probability.
				pred = output.argmax(dim=1, keepdim=True)
				correct += pred.eq(target.view_as(pred)).sum().item()

		accuracy = correct / min(len(valid_loader.dataset), N_VALID_EXAMPLES)

		# attention here
		trial.report(accuracy, epoch)
		# Handle pruning based on the intermediate value.
		if trial.should_prune():
			raise optuna.exceptions.TrialPruned()

	return accuracy


if __name__ == "__main__":
	storage_name = "sqlite:///optuna.db"
	study = optuna.create_study(
		pruner=optuna.pruners.MedianPruner(n_warmup_steps=3), direction="maximize",
		study_name="fashion_mnist_torch", storage=storage_name, load_if_exists=True
	)

	study.optimize(objective, n_trials=20, timeout=1200)

	best_params = study.best_params
	best_value = study.best_value
	print("\n\nbest_value = " + str(best_value))
	print("best_params:")
	print(best_params)




