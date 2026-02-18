#!/usr/bin/env python
# coding: utf-8
"""
Submit experiments for MIRB
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
                    default=1,
                    choices=[1, 3, 10, 30, 100, 300, 1000, 3000, 10000],
                    help="problem size")
parser.add_argument("--penalty",
                    type=float,
                    default=100,
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
config.steepness = 50            # steepness factor
num_blocks = config.size         # number of blocks
train_size = 8000                # number of train
test_size = 100                  # number of test size
val_size = 1000                  # number of validation size

# hyperparameters
hsize_dict = {1:4, 3:8, 10:16, 30:32, 100:64, 300:128, 1000:256, 3000:512, 10000:1024}
config.batch_size = 64                  # batch size
config.hlayers_sol = 5                  # number of hidden layers for solution mapping
config.hlayers_rnd = 4                  # number of hidden layers for solution mapping
config.hsize = hsize_dict[config.size]  # width of hidden layers for solution mapping
config.lr = 1e-3                        # learning rate

# parameters as input data
p_low, p_high = 1.0, 8.0
a_low, a_high = 0.5, 4.5
p_train = torch.empty(train_size, 1).uniform_(p_low, p_high)
p_test  = torch.empty(test_size, 1).uniform_(p_low, p_high)
p_val   = torch.empty(val_size, 1).uniform_(p_low, p_high)
a_train = torch.empty(train_size, num_blocks).uniform_(a_low, a_high)
a_test  = torch.empty(test_size, num_blocks).uniform_(a_low, a_high)
a_val   = torch.empty(val_size, num_blocks).uniform_(a_low, a_high)

# datasets
data_train = DictDataset({"p":p_train, "a":a_train}, name="train")
data_test = DictDataset({"p":p_test, "a":a_test}, name="test")
data_val = DictDataset({"p":p_val, "a":a_val}, name="dev")

# torch dataloaders
loader_train = DataLoader(data_train, config.batch_size, num_workers=0, collate_fn=data_train.collate_fn, shuffle=True, pin_memory=True)
loader_test = DataLoader(data_test, config.batch_size, num_workers=0, collate_fn=data_test.collate_fn, shuffle=False, pin_memory=True)
loader_val = DataLoader(data_val, config.batch_size, num_workers=0, collate_fn=data_val.collate_fn, shuffle=False, pin_memory=True)

print("Rosenbrock")
#if config.project is False:
#    experiments.rosenbrock.run_EX(loader_test, config)
#    experiments.rosenbrock.run_RR(loader_test, config)
#    experiments.rosenbrock.run_N1(loader_test, config)
experiments.rosenbrock.run_AS(loader_train, loader_test, loader_val, config)
experiments.rosenbrock.run_DT(loader_train, loader_test, loader_val, config)
experiments.rosenbrock.run_LR(loader_train, loader_test, loader_val, config)
experiments.rosenbrock.run_RS(loader_train, loader_test, loader_val, config)
