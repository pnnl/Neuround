#!/usr/bin/env python
# coding: utf-8
"""
Submit experiments for MIRB
"""
import argparse
import itertools
import random
import time
import numpy as np
import pandas as pd
import torch
from torch import nn
from tqdm import tqdm

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
parser.add_argument("--warmstart",
                    action="store_true",
                    help="warm start")
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
p_train = np.random.uniform(p_low, p_high, (train_size, 1)).astype(np.float32)
p_test  = np.random.uniform(p_low, p_high, (test_size, 1)).astype(np.float32)
p_val   = np.random.uniform(p_low, p_high, (val_size, 1)).astype(np.float32)
a_train = np.random.uniform(a_low, a_high, (train_size, num_blocks)).astype(np.float32)
a_test  = np.random.uniform(a_low, a_high, (test_size, num_blocks)).astype(np.float32)
a_val   = np.random.uniform(a_low, a_high, (val_size, num_blocks)).astype(np.float32)
# nm datasets
from neuromancer.dataset import DictDataset
data_train = DictDataset({"p":p_train, "a":a_train}, name="train")
data_test = DictDataset({"p":p_test, "a":a_test}, name="test")
data_val = DictDataset({"p":p_val, "a":a_val}, name="dev")
# torch dataloaders
from torch.utils.data import DataLoader
loader_train = DataLoader(data_train, config.batch_size, num_workers=0, collate_fn=data_train.collate_fn, shuffle=True)
loader_test = DataLoader(data_test, config.batch_size, num_workers=0, collate_fn=data_test.collate_fn, shuffle=False)
loader_val = DataLoader(data_val, config.batch_size, num_workers=0, collate_fn=data_val.collate_fn, shuffle=True)

import run
print("Rosenbrock")
if config.project is True and config.warmstart is True:
    print("Warm Starting:")
    run.rosenbrock.exact(loader_test, config)
    run.rosenbrock.rndCls(loader_train, loader_test, loader_val, config)
    run.rosenbrock.rndThd(loader_train, loader_test, loader_val, config)
elif config.project is True:
    print("Feasibility projection:")
    run.rosenbrock.rndCls(loader_train, loader_test, loader_val, config)
    run.rosenbrock.rndThd(loader_train, loader_test, loader_val, config)
    run.rosenbrock.lrnRnd(loader_train, loader_test, loader_val, config)
    run.rosenbrock.rndSte(loader_train, loader_test, loader_val, config)
else:
    run.rosenbrock.exact(loader_test, config)
    run.rosenbrock.relRnd(loader_test, config)
    run.rosenbrock.root(loader_test, config)
    run.rosenbrock.rndCls(loader_train, loader_test, loader_val, config)
    run.rosenbrock.rndThd(loader_train, loader_test, loader_val, config)
    run.rosenbrock.lrnRnd(loader_train, loader_test, loader_val, config)
    run.rosenbrock.rndSte(loader_train, loader_test, loader_val, config)
