#!/usr/bin/env python
# coding: utf-8
"""
Submit experiments for IQP
"""
import argparse
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from neuround import DictDataset
import experiments

# random seed
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# set parser
parser = argparse.ArgumentParser()
parser.add_argument("--size",
                    type=int,
                    default=5,
                    choices=[5, 10, 20, 50, 100, 200, 500, 1000],
                    help="problem size")
parser.add_argument("--penalty",
                    type=float,
                    default=20,
                    help="penalty weight")
parser.add_argument("--project",
                    action="store_true",
                    help="project gradient")
parser.add_argument("--no_compile",
                    action="store_false",
                    dest="compile",
                    help="disable torch.compile")
config = parser.parse_args()

# init problem
num_var = config.size            # number of variables
num_ineq = config.size           # number of constraints
train_size = 8000                # number of train
test_size = 1000                 # number of test size
val_size = 1000                  # number of validation size
num_data = train_size + test_size + val_size

# hyperparameters
hsize_dict = {5:16, 10:32, 20:64, 50:128, 100:256, 200:512, 500:1024, 1000:2048}
config.batch_size = 64                  # batch size
config.hlayers_sol = 5                  # number of hidden layers for solution mapping
config.hlayers_rnd = 4                  # number of hidden layers for solution mapping
config.hsize = hsize_dict[config.size]  # width of hidden layers for solution mapping
config.lr = 1e-3                        # learning rate

# data sample from uniform distribution
b_samples = torch.from_numpy(np.random.uniform(-1, 1, size=(num_data, num_ineq))).float()

# data split
from sklearn.model_selection import train_test_split
from neuround import DictDataset
ind = list(range(num_data))
ind_train, ind_test = train_test_split(ind, test_size=test_size, random_state=42, shuffle=True)
ind_train, ind_val = train_test_split(ind_train, test_size=val_size, random_state=42, shuffle=True)
# Convert indices to tensors for efficient indexing
itrain, itest, ival = torch.tensor(ind_train), torch.tensor(ind_test), torch.tensor(ind_val)
data_train = DictDataset({"b": b_samples[itrain]}, name="train")
data_test = DictDataset({"b": b_samples[itest]}, name="test")
data_val = DictDataset({"b": b_samples[ival]}, name="dev")
# torch dataloaders
from torch.utils.data import DataLoader
loader_train = DataLoader(data_train, config.batch_size, num_workers=0,
                          collate_fn=data_train.collate_fn, shuffle=True, pin_memory=True)
loader_test  = DataLoader(data_test, config.batch_size, num_workers=0,
                          collate_fn=data_test.collate_fn, shuffle=False, pin_memory=True)
loader_val   = DataLoader(data_val, config.batch_size, num_workers=0,
                          collate_fn=data_val.collate_fn, shuffle=False, pin_memory=True)

print("Integer Quadratic")
#if config.project is False:
#    experiments.quadratic.run_EX(loader_test, config)
#    experiments.quadratic.run_RR(loader_test, config)
#    experiments.quadratic.run_N1(loader_test, config)
experiments.quadratic.run_AS(loader_train, loader_test, loader_val, config)
experiments.quadratic.run_DT(loader_train, loader_test, loader_val, config)
experiments.quadratic.run_LR(loader_train, loader_test, loader_val, config)
experiments.quadratic.run_RS(loader_train, loader_test, loader_val, config)
