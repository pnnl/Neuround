#!/usr/bin/env python
# coding: utf-8
"""
Experiment pipeline for Mixed-Integer Rosenbrock (MIRB)
using the neuround API.

This problem has mixed variable types:
  - x[2i]   (even indices) → CONTINUOUS
  - x[2i+1] (odd indices)  → INTEGER

Legacy mapping:
  - nmRosenbrock (custom penaltyLoss) → PenaltyLoss via operator overloading
  - roundGumbelModel                  → StochasticAdaptiveSelectionRounding
  - roundThresholdModel               → DynamicThresholdRounding
  - roundSTEModel                     → StochasticSTERounding
  - netFC                             → MLPBnDrop (dropout=0.2, bnorm=True)
"""

import time
import os
import numpy as np
import pandas as pd
import torch
from torch import nn
from tqdm import tqdm

from neuround import (
    variable, PenaltyLoss, Node, MLPBnDrop, GradientProjection,
    LearnableSolver,
)
from neuround.rounding import (
    StochasticSTERounding,
    DynamicThresholdRounding,
    StochasticAdaptiveSelectionRounding,
)

import logging
logging.getLogger("pyomo.core").setLevel(logging.ERROR)

def _coefficients(num_blocks):
    """Generate fixed linear constraint coefficients (RandomState(17))."""
    rng = np.random.RandomState(17)
    b = torch.from_numpy(rng.normal(scale=1, size=(num_blocks,))).float()
    q = torch.from_numpy(rng.normal(scale=1, size=(num_blocks,))).float()
    return b, q


def build_loss(steepness, num_blocks, penalty_weight, device="cpu"):
    """
    Build PenaltyLoss for the Rosenbrock problem.

    min  sum_i (a_i - x_{2i})^2 + steepness * (x_{2i+1} - x_{2i}^2)^2
    s.t. sum(x_odd) >= num_blocks * p / 2       (inner)
         sum(x_even^2) <= num_blocks * p         (outer)
         b^T x_even <= 0                         (linear 1)
         q^T x_odd <= 0                          (linear 2)
    """
    b_coef, q_coef = _coefficients(num_blocks)
    b_coef, q_coef = b_coef.to(device), q_coef.to(device)
    x = variable("x")
    a = variable("a")
    p = variable("p")
    # slicing: even/odd indices
    x_even = x[:, ::2]   # continuous vars
    x_odd = x[:, 1::2]   # integer vars
    # objective
    f = torch.sum((a - x_even) ** 2 + steepness * (x_odd - x_even ** 2) ** 2, dim=1)
    obj = f.minimize(weight=1.0, name="obj")
    # constraints
    inner = penalty_weight * (num_blocks * p[:, 0:1] / 2 <= torch.sum(x_odd, dim=1, keepdim=True))
    inner.name = "inner"
    outer = penalty_weight * (torch.sum(x_even ** 2, dim=1, keepdim=True) <= num_blocks * p[:, 0:1])
    outer.name = "outer"
    linear1 = penalty_weight * ((x_even @ b_coef) <= 0)
    linear1.name = "linear1"
    linear2 = penalty_weight * ((x_odd @ q_coef) <= 0)
    linear2.name = "linear2"
    return PenaltyLoss(objectives=[obj], constraints=[inner, outer, linear1, linear2])

def run_EX(loader_test, config):
    # Set random seeds
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    # Print experiment info
    print(config)
    print(f"EX in RB for size {config.size}.")
    steepness = config.steepness
    num_blocks = config.size
    # Init exact solver
    from experiments.math_solver.rosenbrock import rosenbrock
    model = rosenbrock(steepness, num_blocks, timelimit=1000)
    # Init result lists
    params, sols, objvals, mean_viols, max_viols, num_viols, elapseds = [], [], [], [], [], [], []
    # Evaluate on test set
    p_test_np = torch.as_tensor(loader_test.dataset.datadict["p"][:100]).cpu().numpy()
    a_test_np = torch.as_tensor(loader_test.dataset.datadict["a"][:100]).cpu().numpy()
    # Loop over test samples
    for i in tqdm(range(100)):
        p, a = p_test_np[i], a_test_np[i]
        # Set parameter values
        model.set_param_val({"p": p, "a": a})
        tick = time.time()
        params.append(p.tolist() + a.tolist())
        # Solve and record results
        try:
            xval, objval = model.solve()
            tock = time.time()
            sols.append(list(list(xval.values())[0].values()))
            objvals.append(objval)
            viol = model.cal_violation()
            mean_viols.append(np.mean(viol))
            max_viols.append(np.max(viol))
            num_viols.append(np.sum(viol > 1e-6))
        except:
            sols.append(None)
            objvals.append(None)
            mean_viols.append(None)
            max_viols.append(None)
            num_viols.append(None)
            tock = time.time()
        # Record elapsed time
        elapseds.append(tock - tick)
    # Create result dataframe and print summary
    os.makedirs("result", exist_ok=True)
    df = pd.DataFrame({"Param": params, "Sol": sols, "Obj Val": objvals,
                        "Mean Violation": mean_viols, "Max Violation": max_viols,
                        "Num Violations": num_viols, "Elapsed Time": elapseds})
    time.sleep(1)
    print(df.describe())
    print("Number of infeasible solutions: {}".format(np.sum(df["Num Violations"] > 0)))
    print("Number of unsolved instances: ", df["Sol"].isna().sum())
    df.to_csv(f"result/rb_exact_{num_blocks}.csv")


def run_RR(loader_test, config):
    # Set random seeds
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    # Print experiment info
    print(config)
    print(f"RR in RB for size {config.size}.")
    from experiments.heuristic import naive_round
    steepness = config.steepness
    num_blocks = config.size
    # Init relaxed solver
    from experiments.math_solver.rosenbrock import rosenbrock
    model = rosenbrock(steepness, num_blocks, timelimit=1000)
    # Init result lists
    params, sols, objvals, mean_viols, max_viols, num_viols, elapseds = [], [], [], [], [], [], []
    # Evaluate on test set
    p_test_np = torch.as_tensor(loader_test.dataset.datadict["p"][:100]).cpu().numpy()
    a_test_np = torch.as_tensor(loader_test.dataset.datadict["a"][:100]).cpu().numpy()
    for i in tqdm(range(100)):
        p, a = p_test_np[i], a_test_np[i]
        # Set parameter values
        model.set_param_val({"p": p, "a": a})
        model_rel = model.relax()
        tick = time.time()
        params.append(p.tolist() + a.tolist())
        # Solve and record results
        try:
            xval_rel, _ = model_rel.solve()
            xval, objval = naive_round(xval_rel, model)
            tock = time.time()
            sols.append(list(list(xval.values())[0].values()))
            objvals.append(objval)
            viol = model.cal_violation()
            mean_viols.append(np.mean(viol))
            max_viols.append(np.max(viol))
            num_viols.append(np.sum(viol > 1e-6))
        except:
            sols.append(None)
            objvals.append(None)
            mean_viols.append(None)
            max_viols.append(None)
            num_viols.append(None)
            tock = time.time()
        # Record elapsed time
        elapseds.append(tock - tick)
    # Create result dataframe and print summary
    os.makedirs("result", exist_ok=True)
    df = pd.DataFrame({"Param": params, "Sol": sols, "Obj Val": objvals,
                        "Mean Violation": mean_viols, "Max Violation": max_viols,
                        "Num Violations": num_viols, "Elapsed Time": elapseds})
    time.sleep(1)
    print(df.describe())
    print("Number of infeasible solutions: {}".format(np.sum(df["Num Violations"] > 0)))
    print("Number of unsolved instances: ", df["Sol"].isna().sum())
    df.to_csv(f"result/rb_rel_{num_blocks}.csv")


def run_N1(loader_test, config):
    # Set random seeds
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    # Print experiment info
    print(config)
    print(f"N1 in RB for size {config.size}.")
    steepness = config.steepness
    num_blocks = config.size
    # Init heuristic solver
    from experiments.math_solver.rosenbrock import rosenbrock
    model = rosenbrock(steepness, num_blocks, timelimit=1000)
    model_heur = model.first_solution_heuristic(nodes_limit=1)
    # Init result lists
    params, sols, objvals, mean_viols, max_viols, num_viols, elapseds = [], [], [], [], [], [], []
    # Evaluate on test set
    p_test_np = torch.as_tensor(loader_test.dataset.datadict["p"][:100]).cpu().numpy()
    a_test_np = torch.as_tensor(loader_test.dataset.datadict["a"][:100]).cpu().numpy()
    for i in tqdm(range(100)):
        p, a = p_test_np[i], a_test_np[i]
        # Set parameter values
        model_heur.set_param_val({"p": p, "a": a})
        tick = time.time()
        params.append(p.tolist() + a.tolist())
        # Solve and record results
        try:
            xval, objval = model_heur.solve()
            tock = time.time()
            sols.append(list(list(xval.values())[0].values()))
            objvals.append(objval)
            viol = model_heur.cal_violation()
            mean_viols.append(np.mean(viol))
            max_viols.append(np.max(viol))
            num_viols.append(np.sum(viol > 1e-6))
        except:
            sols.append(None)
            objvals.append(None)
            mean_viols.append(None)
            max_viols.append(None)
            num_viols.append(None)
            tock = time.time()
        # Record elapsed time
        elapseds.append(tock - tick)
    # Create result dataframe and print summary
    os.makedirs("result", exist_ok=True)
    df = pd.DataFrame({"Param": params, "Sol": sols, "Obj Val": objvals,
                        "Mean Violation": mean_viols, "Max Violation": max_viols,
                        "Num Violations": num_viols, "Elapsed Time": elapseds})
    time.sleep(1)
    print(df.describe())
    print("Number of infeasible solutions: {}".format(np.sum(df["Num Violations"] > 0)))
    print("Number of unsolved instances: ", df["Sol"].isna().sum())
    df.to_csv(f"result/rb_root_{num_blocks}.csv")


def run_AS(loader_train, loader_test, loader_val, config):
    """Adaptive selection rounding (Gumbel)."""
    # Set random seeds
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    # Print experiment info
    print(config)
    print(f"AS in RB for size {config.size}.")
    steepness = config.steepness
    num_blocks = config.size
    hsize = config.hsize
    hlayers_sol = config.hlayers_sol
    hlayers_rnd = config.hlayers_rnd
    lr = config.lr
    penalty_weight = config.penalty
    # Create symbolic variables (mixed types: even → continuous, odd → integer)
    x = variable("x", num_vars=2 * num_blocks,
                 integer_indices=list(range(1, 2 * num_blocks, 2)))
    # Create solution mapping network
    smap_func = MLPBnDrop(insize=num_blocks + 1, outsize=2 * num_blocks,
                          hsizes=[hsize] * hlayers_sol,
                          nonlin=nn.ReLU, dropout=0, bnorm=False)
    smap = Node(smap_func, ["p", "a"], [x.relaxed.key], name="smap")
    # Create rounding network and operator
    rnd_net = MLPBnDrop(insize=3 * num_blocks + 1, outsize=2 * num_blocks,
                        hsizes=[hsize] * hlayers_rnd)
    rnd = StochasticAdaptiveSelectionRounding(
        vars=x, param_keys=["p", "a"], net=rnd_net, continuous_update=True)
    # Build loss
    loss = build_loss(steepness, num_blocks, penalty_weight, device="cuda")
    # Set up solver
    solver = LearnableSolver(smap, rnd, loss)
    # Set up optimizer
    optimizer = torch.optim.AdamW(solver.problem.parameters(), lr=lr)
    # Train
    solver.train(loader_train, loader_val, optimizer, device="cuda", compile=config.compile)
    # Evaluate on test set
    from experiments.math_solver.rosenbrock import rosenbrock
    model = rosenbrock(steepness, num_blocks, timelimit=1000)
    df = evaluate(solver, model, loader_test)
    # Save results
    os.makedirs("result", exist_ok=True)
    if config.project:
        df.to_csv(f"result/rb_cls{penalty_weight}_{num_blocks}-p.csv")
    else:
        df.to_csv(f"result/rb_cls{penalty_weight}_{num_blocks}.csv")


def run_DT(loader_train, loader_test, loader_val, config):
    """Dynamic threshold rounding."""
    # Set random seeds
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    # Print experiment info
    print(config)
    print(f"DT in RB for size {config.size}.")
    steepness = config.steepness
    num_blocks = config.size
    hsize = config.hsize
    hlayers_sol = config.hlayers_sol
    hlayers_rnd = config.hlayers_rnd
    lr = config.lr
    penalty_weight = config.penalty
    # Create symbolic variables
    x = variable("x", num_vars=2 * num_blocks,
                 integer_indices=list(range(1, 2 * num_blocks, 2)))
    # Create solution mapping network
    smap_func = MLPBnDrop(insize=num_blocks + 1, outsize=2 * num_blocks,
                          hsizes=[hsize] * hlayers_sol,
                          nonlin=nn.ReLU, dropout=0, bnorm=False)
    smap = Node(smap_func, ["p", "a"], [x.relaxed.key], name="smap")
    # Create rounding network and operator
    rnd_net = MLPBnDrop(insize=3 * num_blocks + 1, outsize=2 * num_blocks,
                        hsizes=[hsize] * hlayers_rnd)
    rnd = DynamicThresholdRounding(
        vars=x, param_keys=["p", "a"], net=rnd_net, continuous_update=True)
    # Build loss
    loss = build_loss(steepness, num_blocks, penalty_weight, device="cuda")
    # Set up solver
    solver = LearnableSolver(smap, rnd, loss)
    # Set up optimizer
    optimizer = torch.optim.AdamW(solver.problem.parameters(), lr=lr)
    # Train
    solver.train(loader_train, loader_val, optimizer, device="cuda", compile=config.compile)
    # Evaluate on test set
    from experiments.math_solver.rosenbrock import rosenbrock
    model = rosenbrock(steepness, num_blocks, timelimit=1000)
    df = evaluate(solver, model, loader_test)
    # Save results
    os.makedirs("result", exist_ok=True)
    if config.project:
        df.to_csv(f"result/rb_thd{penalty_weight}_{num_blocks}-p.csv")
    else:
        df.to_csv(f"result/rb_thd{penalty_weight}_{num_blocks}.csv")


def run_RS(loader_train, loader_test, loader_val, config):
    """STE rounding."""
    # Set random seeds
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    # Print experiment info
    print(config)
    print(f"STE in RB for size {config.size}.")
    steepness = config.steepness
    num_blocks = config.size
    hsize = config.hsize
    hlayers_sol = config.hlayers_sol
    lr = config.lr
    penalty_weight = config.penalty
    # Create symbolic variables
    x = variable("x", num_vars=2 * num_blocks,
                 integer_indices=list(range(1, 2 * num_blocks, 2)))
    # Create solution mapping network
    smap_func = MLPBnDrop(insize=num_blocks + 1, outsize=2 * num_blocks,
                          hsizes=[hsize] * hlayers_sol,
                          nonlin=nn.ReLU, dropout=0, bnorm=False)
    smap = Node(smap_func, ["p", "a"], [x.relaxed.key], name="smap")
    # Create rounding operator
    rnd = StochasticSTERounding(vars=x)
    # Build loss
    loss = build_loss(steepness, num_blocks, penalty_weight, device="cuda")
    # Set up solver
    solver = LearnableSolver(smap, rnd, loss)
    # Set up optimizer
    optimizer = torch.optim.AdamW(solver.problem.parameters(), lr=lr)
    # Train
    solver.train(loader_train, loader_val, optimizer, device="cuda", compile=config.compile)
    # Evaluate on test set
    from experiments.math_solver.rosenbrock import rosenbrock
    model = rosenbrock(steepness, num_blocks, timelimit=1000)
    df = evaluate(solver, model, loader_test)
    # Save results
    os.makedirs("result", exist_ok=True)
    if config.project:
        df.to_csv(f"result/rb_ste{penalty_weight}_{num_blocks}-p.csv")
    else:
        df.to_csv(f"result/rb_ste{penalty_weight}_{num_blocks}.csv")


def run_LR(loader_train, loader_test, loader_val, config):
    """Learn-then-round: train solution map without rounding, naive round at test."""
    # Set random seeds
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    # Print experiment info
    print(config)
    print(f"LR in RB for size {config.size}.")
    from neuround import Problem, Trainer
    steepness = config.steepness
    num_blocks = config.size
    hsize = config.hsize
    hlayers_sol = config.hlayers_sol
    lr = config.lr
    penalty_weight = config.penalty
    # Create solution mapping network (no rounding layer)
    smap_func = MLPBnDrop(insize=num_blocks + 1, outsize=2 * num_blocks,
                          hsizes=[hsize] * hlayers_sol,
                          nonlin=nn.ReLU, dropout=0, bnorm=False)
    smap = Node(smap_func, ["p", "a"], ["x"], name="smap")
    # Build loss
    loss = build_loss(steepness, num_blocks, penalty_weight, device="cuda")
    # Set up problem and train
    problem = Problem(nodes=[smap], loss=loss)
    problem.to("cuda")
    optimizer = torch.optim.AdamW(problem.parameters(), lr=lr)
    trainer = Trainer(problem, loader_train, loader_val,
                      optimizer=optimizer, device="cuda", 
                      compile=config.compile)
    best_model = trainer.train()
    # Strip "_orig_mod." prefix added by torch.compile if present
    clean = {k.removeprefix("_orig_mod."): v for k, v in best_model.items()}
    problem.load_state_dict(clean)
    # Evaluate with naive rounding
    from experiments.heuristic import naive_round
    from experiments.math_solver.rosenbrock import rosenbrock
    model = rosenbrock(steepness, num_blocks, timelimit=1000)
    params, sols, objvals, mean_viols, max_viols, num_viols, elapseds = [], [], [], [], [], [], []

    # Batch inference for the entire test slice
    p_test_all = torch.as_tensor(loader_test.dataset.datadict["p"][:100]).to("cuda")
    a_test_all = torch.as_tensor(loader_test.dataset.datadict["a"][:100]).to("cuda")
    problem.eval()
    tick_inf = time.time()
    with torch.no_grad():
        test_results = smap({"p": p_test_all, "a": a_test_all})
    tock_inf = time.time()
    # Convert results to numpy for post-processing
    x_all_np = test_results["x"].detach().cpu().numpy()
    p_all_np = p_test_all.detach().cpu().numpy()
    a_all_np = a_test_all.detach().cpu().numpy()
    inf_time_per_sample = (tock_inf - tick_inf) / 100
    # Post-process each sample
    for i in tqdm(range(100)):
        p_np, a_np = p_all_np[i], a_all_np[i]
        model.set_param_val({"p": p_np, "a": a_np})
        for j, val in enumerate(x_all_np[i]):
            model.vars["x"][j].value = val
        xval_rel, _ = model.get_val()
        xval, objval = naive_round(xval_rel, model)
        params.append(p_np.tolist() + a_np.tolist())
        sols.append(list(list(xval.values())[0].values()))
        objvals.append(objval)
        viol = model.cal_violation()
        mean_viols.append(np.mean(viol))
        max_viols.append(np.max(viol))
        num_viols.append(np.sum(viol > 1e-6))
        elapseds.append(inf_time_per_sample)
    # Create result dataframe and print summary
    df = pd.DataFrame({"Param": params, "Sol": sols, "Obj Val": objvals,
                        "Mean Violation": mean_viols, "Max Violation": max_viols,
                        "Num Violations": num_viols, "Elapsed Time": elapseds})
    time.sleep(1)
    os.makedirs("result", exist_ok=True)
    print(df.describe())
    print("Number of infeasible solutions: {}".format(np.sum(df["Num Violations"] > 0)))
    df.to_csv(f"result/rb_lrn{config.penalty}_{num_blocks}.csv")


def evaluate(solver, model, loader_test):
    """Evaluate a LearnableSolver against the math solver ground truth."""
    # Initialize result lists
    params, sols, objvals, mean_viols, max_viols, num_viols, elapseds = [], [], [], [], [], [], []
    # Evaluate on test set
    p_test_all = torch.as_tensor(loader_test.dataset.datadict["p"][:100]).to("cuda")
    a_test_all = torch.as_tensor(loader_test.dataset.datadict["a"][:100]).to("cuda")
    tick_inf = time.time()
    test_results = solver.predict({"p": p_test_all, "a": a_test_all})
    tock_inf = time.time()
    # Convert results to numpy for post-processing
    x_all_np = test_results["x"].detach().cpu().numpy()
    p_all_np = p_test_all.detach().cpu().numpy()
    a_all_np = a_test_all.detach().cpu().numpy()
    inf_time_per_sample = (tock_inf - tick_inf) / 100
    # Post-process each sample
    for i in tqdm(range(100)):
        p_np, a_np = p_all_np[i], a_all_np[i]
        model.set_param_val({"p": p_np, "a": a_np})
        for j, val in enumerate(x_all_np[i]):
            model.vars["x"][j].value = val
        # Get solution and objective value
        xval, objval = model.get_val()
        # Record results
        params.append(p_np.tolist() + a_np.tolist())
        sols.append(list(list(xval.values())[0].values()))
        objvals.append(objval)
        viol = model.cal_violation()
        mean_viols.append(np.mean(viol))
        max_viols.append(np.max(viol))
        num_viols.append(np.sum(viol > 1e-6))
        elapseds.append(inf_time_per_sample)
    # Create result dataframe and print summary
    df = pd.DataFrame({"Param": params, "Sol": sols, "Obj Val": objvals,
                        "Mean Violation": mean_viols, "Max Violation": max_viols,
                        "Num Violations": num_viols, "Elapsed Time": elapseds})
    print(df.describe())
    print("Number of infeasible solutions: {}".format(np.sum(df["Num Violations"] > 0)))
    return df
