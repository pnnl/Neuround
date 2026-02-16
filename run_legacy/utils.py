#!/usr/bin/env python
# coding: utf-8
"""
Utlities
"""

import torch
from src.problem.neuromancer.trainer import trainer

def train(components, loss_fn, loader_train, loader_val, lr, penalty_growth):
    epochs = 200                    # number of training epochs
    patience = 20                   # number of epochs with no improvement in eval metric to allow before early stopping
    if penalty_growth:
        growth_rate = 1.05          # growth rate of penalty weight
        warmup = 50                 # number of epochs to wait before enacting early stopping policies
    else:
        growth_rate = 1             # growth rate of penalty weight
        warmup = 20                 # number of epochs to wait before enacting early stopping policies
    optimizer = torch.optim.AdamW(components.parameters(), lr=lr)
    # create a trainer for the problem
    my_trainer = trainer(components, loss_fn, optimizer, epochs=epochs,
                         growth_rate=growth_rate, patience=patience, warmup=warmup,
                         device="cuda")
    # training for the rounding problem
    my_trainer.train(loader_train, loader_val)
