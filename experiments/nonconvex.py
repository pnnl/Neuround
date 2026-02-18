#!/usr/bin/env python
# coding: utf-8
"""
Experiment pipeline for Integer Nonconvex Problem (INC)
using the neuround API.

Legacy mapping:
  - nmNonconvex (custom penaltyLoss)   → PenaltyLoss via operator overloading
  - roundGumbelModel                   → StochasticAdaptiveSelectionRounding
  - roundThresholdModel                → DynamicThresholdRounding
  - roundSTEModel                      → StochasticSTERounding
  - netFC                              → MLPBnDrop (dropout=0.2, bnorm=True)
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

def _coefficients(num_var, num_ineq):
    """Generate fixed problem coefficients (RandomState(17))."""
    rng = np.random.RandomState(17)
    Q = torch.from_numpy(0.01 * np.diag(rng.random(size=num_var))).float()
    p = torch.from_numpy(0.1 * rng.random(num_var)).float()
    A = torch.from_numpy(rng.normal(scale=0.1, size=(num_ineq, num_var))).float()
    return Q, p, A


def build_loss(num_var, num_ineq, penalty_weight, device="cpu"):
    """
    Build PenaltyLoss for the nonconvex problem.

    min  (1/2) x^T Q x + p^T sin(x)
    s.t. A(d)x <= b  where A(d) perturbs col 0 by +d, col 1 by -d
    """
    Q, p, A = _coefficients(num_var, num_ineq)
    Q, p, A = Q.to(device), p.to(device), A.to(device)
    x = variable("x")
    b = variable("b")
    d = variable("d")
    # objective (nonconvex due to sin)
    f = 0.5 * torch.sum((x @ Q) * x, dim=1) + torch.sum(p * torch.sin(x), dim=1)
    obj = f.minimize(weight=1.0, name="obj")
    # parameterized constraint: A(d)x = Ax + d*(x_0 - x_1) <= b
    lhs = x @ A.T + d * (x[:, [0]] - x[:, [1]])
    con = penalty_weight * (lhs <= b)
    con.name = "con"
    return PenaltyLoss(objectives=[obj], constraints=[con])


def run_EX(loader_test, config):
    # Set random seeds
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    # Print experiment info
    print(config)
    print(f"EX in NC for size {config.size}.")
    num_var = config.size
    num_ineq = config.size
    # Init exact solver
    from experiments.math_solver.nonconvex import nonconvex
    model = nonconvex(num_var, num_ineq, timelimit=1000)
    # Init result lists
    params, sols, objvals, mean_viols, max_viols, num_viols, elapseds = [], [], [], [], [], [], []
    # Evaluate on test set
    b_test_np = torch.as_tensor(loader_test.dataset.datadict["b"][:100]).cpu().numpy()
    d_test_np = torch.as_tensor(loader_test.dataset.datadict["d"][:100]).cpu().numpy()
    # Loop over test samples
    for i in tqdm(range(100)):
        b, d = b_test_np[i], d_test_np[i]
        # Set parameter values
        model.set_param_val({"b": b, "d": d})
        tick = time.time()
        params.append(b.tolist() + d.tolist())
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
    df = pd.DataFrame({"Param": params, "Sol": sols, "Obj Val": objvals,
                        "Mean Violation": mean_viols, "Max Violation": max_viols,
                        "Num Violations": num_viols, "Elapsed Time": elapseds})
    time.sleep(1)
    print(df.describe())
    print("Number of infeasible solutions: {}".format(np.sum(df["Num Violations"] > 0)))
    print("Number of unsolved instances: ", df["Sol"].isna().sum())
    os.makedirs("result", exist_ok=True)
    df.to_csv(f"result/nc_exact_{num_var}-{num_ineq}.csv")


def run_RR(loader_test, config):
    # Set random seeds
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    # Print experiment info
    print(config)
    print(f"RR in NC for size {config.size}.")
    from experiments.heuristic import naive_round
    num_var = config.size
    num_ineq = config.size
    # Init relaxed solver
    from experiments.math_solver.nonconvex import nonconvex
    model = nonconvex(num_var, num_ineq, timelimit=1000)
    # Init result lists
    params, sols, objvals, mean_viols, max_viols, num_viols, elapseds = [], [], [], [], [], [], []
    # Evaluate on test set
    b_test_np = torch.as_tensor(loader_test.dataset.datadict["b"][:100]).cpu().numpy()
    d_test_np = torch.as_tensor(loader_test.dataset.datadict["d"][:100]).cpu().numpy()

    for i in tqdm(range(100)):
        b, d = b_test_np[i], d_test_np[i]
        # Set parameter values
        model.set_param_val({"b": b})
        model_rel = model.relax()
        tick = time.time()
        params.append(b.tolist() + d.tolist())
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
    df = pd.DataFrame({"Param": params, "Sol": sols, "Obj Val": objvals,
                        "Mean Violation": mean_viols, "Max Violation": max_viols,
                        "Num Violations": num_viols, "Elapsed Time": elapseds})
    time.sleep(1)
    print(df.describe())
    print("Number of infeasible solutions: {}".format(np.sum(df["Num Violations"] > 0)))
    print("Number of unsolved instances: ", df["Sol"].isna().sum())
    os.makedirs("result", exist_ok=True)
    df.to_csv(f"result/nc_rel_{num_var}-{num_ineq}.csv")


def run_N1(loader_test, config):
    # Set random seeds
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    # Print experiment info
    print(config)
    print(f"N1 in NC for size {config.size}.")
    num_var = config.size
    num_ineq = config.size
    # Init heuristic solver
    from experiments.math_solver.nonconvex import nonconvex
    model = nonconvex(num_var, num_ineq, timelimit=1000)
    model_heur = model.first_solution_heuristic(nodes_limit=1)
    # Init result lists
    params, sols, objvals, mean_viols, max_viols, num_viols, elapseds = [], [], [], [], [], [], []
    # Evaluate on test set
    b_test_np = torch.as_tensor(loader_test.dataset.datadict["b"][:100]).cpu().numpy()
    d_test_np = torch.as_tensor(loader_test.dataset.datadict["d"][:100]).cpu().numpy()

    for i in tqdm(range(100)):
        b, d = b_test_np[i], d_test_np[i]
        # Set parameter values
        model_heur.set_param_val({"b": b, "d": d})
        tick = time.time()
        params.append(b.tolist() + d.tolist())
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
    df = pd.DataFrame({"Param": params, "Sol": sols, "Obj Val": objvals,
                        "Mean Violation": mean_viols, "Max Violation": max_viols,
                        "Num Violations": num_viols, "Elapsed Time": elapseds})
    time.sleep(1)
    print(df.describe())
    print("Number of infeasible solutions: {}".format(np.sum(df["Num Violations"] > 0)))
    print("Number of unsolved instances: ", df["Sol"].isna().sum())
    os.makedirs("result", exist_ok=True)
    df.to_csv(f"result/nc_root_{num_var}-{num_ineq}.csv")


def run_AS(loader_train, loader_test, loader_val, config):
    """Adaptive selection rounding (Gumbel)."""
    # Set random seeds
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    # Print experiment info
    print(config)
    print(f"AS in NC for size {config.size}.")
    num_var = config.size
    num_ineq = config.size
    hsize = config.hsize
    hlayers_sol = config.hlayers_sol
    hlayers_rnd = config.hlayers_rnd
    lr = config.lr
    penalty_weight = config.penalty
    # Create symbolic variables
    x = variable("x", num_vars=num_var, integer_indices=list(range(num_var)))
    # Create solution mapping network
    smap_func = MLPBnDrop(insize=num_ineq * 2, outsize=num_var,
                          hsizes=[hsize] * hlayers_sol,
                          nonlin=nn.ReLU, dropout=0, bnorm=False)
    smap = Node(smap_func, ["b", "d"], [x.relaxed.key], name="smap")
    # Create rounding network and operator
    rnd_net = MLPBnDrop(insize=num_ineq * 2 + num_var, outsize=num_var,
                        hsizes=[hsize] * hlayers_rnd)
    rnd = StochasticAdaptiveSelectionRounding(
        vars=x, param_keys=["b", "d"], net=rnd_net, continuous_update=True)
    # Build loss
    loss = build_loss(num_var, num_ineq, penalty_weight, device="cuda")
    # Set up solver
    solver = LearnableSolver(smap, rnd, loss)
    # Set up optimizer
    optimizer = torch.optim.AdamW(solver.problem.parameters(), lr=lr)
    # Train
    solver.train(loader_train, loader_val, optimizer, device="cuda", compile=config.compile)
    # Evaluate on test set
    from experiments.math_solver.nonconvex import nonconvex
    model = nonconvex(num_var, num_ineq, timelimit=1000)
    df = evaluate(solver, model, loader_test)
    # Save results
    os.makedirs("result", exist_ok=True)
    if config.project:
        df.to_csv(f"result/nc_cls{penalty_weight}_{num_var}-{num_ineq}-p.csv")
    else:
        df.to_csv(f"result/nc_cls{penalty_weight}_{num_var}-{num_ineq}.csv")


def run_DT(loader_train, loader_test, loader_val, config):
    """Dynamic threshold rounding."""
    # Set random seeds
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    # Print experiment info
    print(config)
    print(f"DT in NC for size {config.size}.")
    num_var = config.size
    num_ineq = config.size
    hsize = config.hsize
    hlayers_sol = config.hlayers_sol
    hlayers_rnd = config.hlayers_rnd
    lr = config.lr
    penalty_weight = config.penalty
    # Create symbolic variables
    x = variable("x", num_vars=num_var, integer_indices=list(range(num_var)))
    # Create solution mapping network
    smap_func = MLPBnDrop(insize=num_ineq * 2, outsize=num_var,
                          hsizes=[hsize] * hlayers_sol,
                          nonlin=nn.ReLU, dropout=0, bnorm=False)
    smap = Node(smap_func, ["b", "d"], [x.relaxed.key], name="smap")
    # Create rounding network and operator
    rnd_net = MLPBnDrop(insize=num_ineq * 2 + num_var, outsize=num_var,
                        hsizes=[hsize] * hlayers_rnd)
    rnd = DynamicThresholdRounding(
        vars=x, param_keys=["b", "d"], net=rnd_net, continuous_update=True)
    # Build loss
    loss = build_loss(num_var, num_ineq, penalty_weight, device="cuda")
    # Set up solver
    solver = LearnableSolver(smap, rnd, loss)
    # Set up optimizer
    optimizer = torch.optim.AdamW(solver.problem.parameters(), lr=lr)
    # Train
    solver.train(loader_train, loader_val, optimizer, device="cuda", compile=config.compile)
    # Evaluate on test set
    from experiments.math_solver.nonconvex import nonconvex
    model = nonconvex(num_var, num_ineq, timelimit=1000)
    df = evaluate(solver, model, loader_test)
    # Save results
    os.makedirs("result", exist_ok=True)
    if config.project:
        df.to_csv(f"result/nc_thd{penalty_weight}_{num_var}-{num_ineq}-p.csv")
    else:
        df.to_csv(f"result/nc_thd{penalty_weight}_{num_var}-{num_ineq}.csv")


def run_RS(loader_train, loader_test, loader_val, config):
    """STE rounding."""
    # Set random seeds
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    # Print experiment info
    print(config)
    print(f"STE in NC for size {config.size}.")
    num_var = config.size
    num_ineq = config.size
    hsize = config.hsize
    hlayers_sol = config.hlayers_sol
    lr = config.lr
    penalty_weight = config.penalty
    # Create symbolic variables
    x = variable("x", num_vars=num_var, integer_indices=list(range(num_var)))
    # Create solution mapping network
    smap_func = MLPBnDrop(insize=num_ineq * 2, outsize=num_var,
                          hsizes=[hsize] * hlayers_sol,
                          nonlin=nn.ReLU, dropout=0, bnorm=False)
    smap = Node(smap_func, ["b", "d"], [x.relaxed.key], name="smap")
    # Create rounding operator
    rnd = StochasticSTERounding(vars=x)
    # Build loss
    loss = build_loss(num_var, num_ineq, penalty_weight, device="cuda")
    # Set up solver
    solver = LearnableSolver(smap, rnd, loss)
    # Set up optimizer
    optimizer = torch.optim.AdamW(solver.problem.parameters(), lr=lr)
    # Train
    solver.train(loader_train, loader_val, optimizer, device="cuda", compile=config.compile)
    # Evaluate on test set
    from experiments.math_solver.nonconvex import nonconvex
    model = nonconvex(num_var, num_ineq, timelimit=1000)
    df = evaluate(solver, model, loader_test)
    # Save results
    os.makedirs("result", exist_ok=True)
    if config.project:
        df.to_csv(f"result/nc_ste{penalty_weight}_{num_var}-{num_ineq}-p.csv")
    else:
        df.to_csv(f"result/nc_ste{penalty_weight}_{num_var}-{num_ineq}.csv")


def run_LR(loader_train, loader_test, loader_val, config):
    """Learn-then-round: train solution map without rounding, naive round at test."""
    # Set random seeds
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    # Print experiment info
    print(config)
    print(f"LR in NC for size {config.size}.")
    from neuround import Problem, Trainer
    num_var = config.size
    num_ineq = config.size
    hsize = config.hsize
    hlayers_sol = config.hlayers_sol
    lr = config.lr
    penalty_weight = config.penalty
    # Create solution mapping network (no rounding layer)
    smap_func = MLPBnDrop(insize=num_ineq * 2, outsize=num_var,
                          hsizes=[hsize] * hlayers_sol,
                          nonlin=nn.ReLU, dropout=0, bnorm=False)
    smap = Node(smap_func, ["b", "d"], ["x"], name="smap")
    # Build loss
    loss = build_loss(num_var, num_ineq, penalty_weight, device="cuda")
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
    from experiments.math_solver.nonconvex import nonconvex
    model = nonconvex(num_var, num_ineq, timelimit=1000)
    params, sols, objvals, mean_viols, max_viols, num_viols, elapseds = [], [], [], [], [], [], []

    # Batch inference for the entire test slice
    b_test_all = torch.as_tensor(loader_test.dataset.datadict["b"][:100]).to("cuda")
    d_test_all = torch.as_tensor(loader_test.dataset.datadict["d"][:100]).to("cuda")
    problem.eval()
    tick_inf = time.time()
    with torch.no_grad():
        test_results = smap({"b": b_test_all, "d": d_test_all})
    tock_inf = time.time()
    # Convert results to numpy for post-processing
    x_all_np = test_results["x"].detach().cpu().numpy()
    b_all_np = b_test_all.detach().cpu().numpy()
    d_all_np = d_test_all.detach().cpu().numpy()
    inf_time_per_sample = (tock_inf - tick_inf) / 100
    # Post-process each sample
    for i in tqdm(range(100)):
        b_np, d_np = b_all_np[i], d_all_np[i]
        model.set_param_val({"b": b_np, "d": d_np})
        for j, val in enumerate(x_all_np[i]):
            model.vars["x"][j].value = val
        xval_rel, _ = model.get_val()
        xval, objval = naive_round(xval_rel, model)
        params.append(b_np.tolist() + d_np.tolist())
        sols.append(list(list(xval.values())[0].values()))
        objvals.append(objval)
        viol = model.cal_violation()
        mean_viols.append(np.mean(viol))
        max_viols.append(np.max(viol))
        num_viols.append(np.sum(viol > 1e-6))
        elapseds.append(inf_time_per_sample)
    df = pd.DataFrame({"Param": params, "Sol": sols, "Obj Val": objvals,
                        "Mean Violation": mean_viols, "Max Violation": max_viols,
                        "Num Violations": num_viols, "Elapsed Time": elapseds})
    time.sleep(1)
    os.makedirs("result", exist_ok=True)
    print(df.describe())
    print("Number of infeasible solutions: {}".format(np.sum(df["Num Violations"] > 0)))
    df.to_csv(f"result/nc_lrn{config.penalty}_{num_var}-{num_ineq}.csv")


def evaluate(solver, model, loader_test):
    """Evaluate a LearnableSolver against the math solver ground truth."""
    # Initialize result lists
    params, sols, objvals, mean_viols, max_viols, num_viols, elapseds = [], [], [], [], [], [], []
    # Evaluate on test set
    b_test_all = torch.as_tensor(loader_test.dataset.datadict["b"][:100]).to("cuda")
    d_test_all = torch.as_tensor(loader_test.dataset.datadict["d"][:100]).to("cuda")
    # Batch inference for the entire test slice
    tick_inf = time.time()
    test_results = solver.predict({"b": b_test_all, "d": d_test_all})
    tock_inf = time.time()
    # Convert results to numpy for post-processing
    x_all_np = test_results["x"].detach().cpu().numpy()
    b_all_np = b_test_all.detach().cpu().numpy()
    d_all_np = d_test_all.detach().cpu().numpy()
    inf_time_per_sample = (tock_inf - tick_inf) / 100
    # Post-process each sample
    for i in tqdm(range(100)):
        b_np, d_np = b_all_np[i], d_all_np[i]
        model.set_param_val({"b": b_np, "d": d_np})
        for j, val in enumerate(x_all_np[i]):
            model.vars["x"][j].value = val
        # Get solution and objective value
        xval, objval = model.get_val()
        # Record results
        params.append(b_np.tolist() + d_np.tolist())
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
