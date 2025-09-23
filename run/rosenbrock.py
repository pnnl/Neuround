#!/usr/bin/env python
# coding: utf-8
"""
Experiment pipeline for QP
"""
import random
import time
import numpy as np
import pandas as pd
import torch
from torch import nn
from tqdm import tqdm
from run import utils
import gurobipy as gp
import os
import ast
import time
import io
import sys

# turn off warning
import logging
logging.getLogger("pyomo.core").setLevel(logging.ERROR)

def exact(loader_test, config):
    print(config)
    # random seed
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    print(f"EX in RB for size {config.size}.")
    # config
    steepness = config.steepness
    num_blocks = config.size
    # init model
    from src.problem import msRosenbrock
    model = msRosenbrock(steepness, num_blocks, timelimit=1000)
    # init df
    params, sols, objvals, mean_viols, max_viols, num_viols, elapseds = [], [], [], [], [], [], []
    # go through test data
    p_test = loader_test.dataset.datadict["p"]
    a_test = loader_test.dataset.datadict["a"]
    for p, a in tqdm(list(zip(p_test, a_test))):
        # set params
        model.set_param_val({"p":p, "a":a})
        # solve
        tick = time.time()
        params.append(list(p)+list(a))
        try:
            xval, objval = model.solve("scip")
            tock = time.time()
            # eval
            sols.append(list(list(xval.values())[0].values()))
            objvals.append(objval)
            viol = model.cal_violation()
            mean_viols.append(np.mean(viol))
            max_viols.append(np.max(viol))
            num_viols.append(np.sum(viol > 1e-6))
        except:
            # infeasible
            sols.append(None)
            objvals.append(None)
            mean_viols.append(None)
            max_viols.append(None)
            num_viols.append(None)
            tock = time.time()
        elapseds.append(tock - tick)
    df = pd.DataFrame({"Param": params,
                       "Sol": sols,
                       "Obj Val": objvals,
                       "Mean Violation": mean_viols,
                       "Max Violation": max_viols,
                       "Num Violations": num_viols,
                       "Elapsed Time": elapseds})
    time.sleep(1)
    print(df.describe())
    print("Number of infeasible solutions: {}".format(np.sum(df["Num Violations"] > 0)))
    print("Number of unsolved instances: ", df["Sol"].isna().sum())
    df.to_csv(f"result/rb_exact_{num_blocks}.csv")


def relRnd(loader_test, config):
    print(config)
    # random seed
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    print(f"RR in RB for size {config.size}.")
    from src.heuristic import naive_round
    # config
    steepness = config.steepness
    num_blocks = config.size
    # init model
    from src.problem import msRosenbrock
    model = msRosenbrock(steepness, num_blocks, timelimit=1000)
    # init df
    params, sols, objvals, mean_viols, max_viols, num_viols, elapseds = [], [], [], [], [], [], []
    # go through test data
    p_test = loader_test.dataset.datadict["p"]
    a_test = loader_test.dataset.datadict["a"]
    for p, a in tqdm(list(zip(p_test, a_test))):
        # set params
        model.set_param_val({"p":p, "a":a})
        model_rel = model.relax()
        # solve
        tick = time.time()
        params.append(list(p)+list(a))
        try:
            xval_rel, _ = model_rel.solve("guorbi")
            xval, objval = naive_round(xval_rel, model)
            tock = time.time()
            # eval
            sols.append(list(list(xval.values())[0].values()))
            objvals.append(objval)
            viol = model.cal_violation()
            mean_viols.append(np.mean(viol))
            max_viols.append(np.max(viol))
            num_viols.append(np.sum(viol > 1e-6))
        except:
            # infeasible
            sols.append(None)
            objvals.append(None)
            mean_viols.append(None)
            max_viols.append(None)
            num_viols.append(None)
            tock = time.time()
        elapseds.append(tock - tick)
    df = pd.DataFrame({"Param": params,
                       "Sol": sols,
                       "Obj Val": objvals,
                       "Mean Violation": mean_viols,
                       "Max Violation": max_viols,
                       "Num Violations": num_viols,
                       "Elapsed Time": elapseds})
    time.sleep(1)
    print(df.describe())
    print("Number of infeasible solutions: {}".format(np.sum(df["Num Violations"] > 0)))
    print("Number of unsolved instances: ", df["Sol"].isna().sum())
    df.to_csv(f"result/rb_rel_{num_blocks}.csv")


def root(loader_test, config):
    print(config)
    # random seed
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    print(f"N1 in RB for size {config.size}.")
    # config
    steepness = config.steepness
    num_blocks = config.size
    # init model
    # init model
    from src.problem import msRosenbrock
    model = msRosenbrock(steepness, num_blocks, timelimit=1000)
    model_heur = model.first_solution_heuristic(nodes_limit=1)
    # init df
    params, sols, objvals, mean_viols, max_viols, num_viols, elapseds = [], [], [], [], [], [], []
    # go through test data
    p_test = loader_test.dataset.datadict["p"]
    a_test = loader_test.dataset.datadict["a"]
    for p, a in tqdm(list(zip(p_test, a_test))):
        # set params
        model_heur.set_param_val({"p":p, "a":a})
        # solve
        tick = time.time()
        params.append(list(p)+list(a))
        try:
            xval, objval = model_heur.solve("guorbi")
            tock = time.time()
            # eval
            sols.append(list(list(xval.values())[0].values()))
            objvals.append(objval)
            viol = model_heur.cal_violation()
            mean_viols.append(np.mean(viol))
            max_viols.append(np.max(viol))
            num_viols.append(np.sum(viol > 1e-6))
        except:
            # infeasible
            sols.append(None)
            objvals.append(None)
            mean_viols.append(None)
            max_viols.append(None)
            num_viols.append(None)
            tock = time.time()
        elapseds.append(tock - tick)
    df = pd.DataFrame({"Param": params,
                       "Sol": sols,
                       "Obj Val": objvals,
                       "Mean Violation": mean_viols,
                       "Max Violation": max_viols,
                       "Num Violations": num_viols,
                       "Elapsed Time": elapseds})
    time.sleep(1)
    print(df.describe())
    print("Number of infeasible solutions: {}".format(np.sum(df["Num Violations"] > 0)))
    print("Number of unsolved instances: ", df["Sol"].isna().sum())
    df.to_csv(f"result/rb_root_{num_blocks}.csv")


def rndCls(loader_train, loader_test, loader_val, config, penalty_growth=False):
    print(config)
    # random seed
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    print(f"RC in RB for size {config.size}.")
    import neuromancer as nm
    from src.problem import nmRosenbrock
    from src.func.layer import netFC
    from src.func import roundGumbelModel
    # config
    steepness = config.steepness
    num_blocks = config.size
    hlayers_sol = config.hlayers_sol
    hlayers_rnd = config.hlayers_rnd
    hsize = config.hsize
    lr = config.lr
    penalty_weight = config.penalty
    project = config.project
    # init model
    from src.problem import msRosenbrock
    model = msRosenbrock(steepness, num_blocks, timelimit=1000)
    # build neural architecture for the solution map
    func = nm.modules.blocks.MLP(insize=num_blocks+1, outsize=2*num_blocks, bias=True,
                                 linear_map=nm.slim.maps["linear"],
                                 nonlin=nn.ReLU, hsizes=[hsize]*hlayers_sol)
    smap = nm.system.Node(func, ["p", "a"], ["x"], name="smap")
    # define rounding model
    layers_rnd = netFC(input_dim=3*num_blocks+1, hidden_dims=[hsize]*hlayers_rnd, output_dim=2*num_blocks)
    rnd = roundGumbelModel(layers=layers_rnd, param_keys=["p", "a"], var_keys=["x"], output_keys=["x_rnd"],
                           int_ind=model.int_ind, continuous_update=True, name="round")
    # build neuromancer problem for rounding
    components = nn.ModuleList([smap, rnd]).to("cuda")
    loss_fn = nmRosenbrock(["p", "a", "x_rnd"], steepness, num_blocks, penalty_weight)
    # train
    utils.train(components, loss_fn, loader_train, loader_val, lr, penalty_growth)
    # eval
    df = evaluate(components, loss_fn, model, loader_test, project)
    if penalty_growth:
        df.to_csv(f"result/rb_cls{penalty_weight}_{num_blocks}-g.csv")
    elif config.samples == 800:
        df.to_csv(f"result/rb_cls{penalty_weight}_{num_blocks}-s.csv")
    elif config.samples == 80000:
        df.to_csv(f"result/rb_cls{penalty_weight}_{num_blocks}-l.csv")
    elif config.project:
        df.to_csv(f"result/rb_cls{penalty_weight}_{num_blocks}-p.csv")
    elif config.project and config.samples == 800:
        df.to_csv(f"result/rb_cls{penalty_weight}_{num_blocks}-s-p.csv")
    elif config.project and config.samples == 80000:
        df.to_csv(f"result/rb_cls{penalty_weight}_{num_blocks}-l-p.csv")
    else:
        df.to_csv(f"result/rb_cls{penalty_weight}_{num_blocks}.csv")

def rndThd(loader_train, loader_test, loader_val, config, penalty_growth=False):
    print(config)
    # random seed
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    print(f"LT in RB for size {config.size}.")
    import neuromancer as nm
    from src.problem import nmRosenbrock
    from src.func.layer import netFC
    from src.func import roundThresholdModel
    steepness = config.steepness
    num_blocks = config.size
    hlayers_sol = config.hlayers_sol
    hlayers_rnd = config.hlayers_rnd
    hsize = config.hsize
    lr = config.lr
    penalty_weight = config.penalty
    project = config.project
    # init model
    from src.problem import msRosenbrock
    model = msRosenbrock(steepness, num_blocks, timelimit=1000)
    # build neural architecture for the solution map
    func = nm.modules.blocks.MLP(insize=num_blocks+1, outsize=2*num_blocks, bias=True,
                                 linear_map=nm.slim.maps["linear"],
                                 nonlin=nn.ReLU, hsizes=[hsize]*hlayers_sol)
    smap = nm.system.Node(func, ["p", "a"], ["x"], name="smap")
    # define rounding model
    layers_rnd = netFC(input_dim=3*num_blocks+1, hidden_dims=[hsize]*hlayers_rnd, output_dim=2*num_blocks)
    rnd = roundThresholdModel(layers=layers_rnd, param_keys=["p", "a"], var_keys=["x"],  output_keys=["x_rnd"],
                              int_ind=model.int_ind, continuous_update=True, name="round")
    # build neuromancer problem for rounding
    components = nn.ModuleList([smap, rnd]).to("cuda")
    loss_fn = nmRosenbrock(["p", "a", "x_rnd"], steepness, num_blocks, penalty_weight)
    # train
    utils.train(components, loss_fn, loader_train, loader_val, lr, penalty_growth)
    # eval
    df = evaluate(components, loss_fn, model, loader_test, project)
    if penalty_growth:
        df.to_csv(f"result/rb_thd{penalty_weight}_{num_blocks}-g.csv")
    elif config.samples == 800:
        df.to_csv(f"result/rb_thd{penalty_weight}_{num_blocks}-s.csv")
    elif config.samples == 80000:
        df.to_csv(f"result/rb_thd{penalty_weight}_{num_blocks}-l.csv")
    elif config.project:
        df.to_csv(f"result/rb_thd{penalty_weight}_{num_blocks}-p.csv")
    elif config.project and config.samples == 800:
        df.to_csv(f"result/rb_thd{penalty_weight}_{num_blocks}-s-p.csv")
    elif config.project and config.samples == 80000:
        df.to_csv(f"result/rb_thd{penalty_weight}_{num_blocks}-l-p.csv")
    else:
        df.to_csv(f"result/rb_thd{penalty_weight}_{num_blocks}.csv")

def lrnRnd(loader_train, loader_test, loader_val, config, penalty_growth=False):
    print(config)
    # random seed
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    print(f"RL in RB for size {config.size}.")
    import neuromancer as nm
    from src.problem import nmRosenbrock
    from src.func.layer import netFC
    from src.func import roundThresholdModel
    # config
    steepness = config.steepness
    num_blocks = config.size
    hlayers_sol = config.hlayers_sol
    hsize = config.hsize
    lr = config.lr
    penalty_weight = config.penalty
    # init model
    from src.problem import msRosenbrock
    model = msRosenbrock(steepness, num_blocks, timelimit=1000)
    # build neural architecture for the solution map
    func = nm.modules.blocks.MLP(insize=num_blocks+1, outsize=2*num_blocks, bias=True,
                                 linear_map=nm.slim.maps["linear"],
                                 nonlin=nn.ReLU, hsizes=[hsize]*hlayers_sol)
    smap = nm.system.Node(func, ["p", "a"], ["x"], name="smap")
    # build neuromancer problem for rounding
    components = nn.ModuleList([smap]).to("cuda")
    loss_fn = nmRosenbrock(["p", "a", "x"], steepness, num_blocks, penalty_weight)
    # train
    utils.train(components, loss_fn, loader_train, loader_val, lr, penalty_growth)
    # eval
    from src.heuristic import naive_round
    params, sols, objvals, mean_viols, max_viols, num_viols, elapseds = [], [], [], [], [], [], []
    p_test = loader_test.dataset.datadict["p"]
    a_test = loader_test.dataset.datadict["a"]
    for p, a in tqdm(list(zip(p_test, a_test))):
        # data point as tensor
        datapoints = {"p": torch.tensor(np.array([p]), dtype=torch.float32).to("cuda"),
                      "a": torch.tensor(np.array([a]), dtype=torch.float32).to("cuda"),
                      "name": "test"}
        # infer
        components.eval()
        tick = time.time()
        with torch.no_grad():
            for comp in components:
                datapoints.update(comp(datapoints))
        tock = time.time()
        # assign params
        model.set_param_val({"p":p, "a":a})
        # assign vars
        x = datapoints["x"]
        for i in range(num_blocks*2):
            model.vars["x"][i].value = x[0,i].item()
        # get solutions
        xval_rel, _ = model.get_val()
        xval, objval = naive_round(xval_rel, model)
        params.append(list(p)+list(a))
        sols.append(list(list(xval.values())[0].values()))
        objvals.append(objval)
        viol = model.cal_violation()
        mean_viols.append(np.mean(viol))
        max_viols.append(np.max(viol))
        num_viols.append(np.sum(viol > 1e-6))
        elapseds.append(tock - tick)
    df = pd.DataFrame({"Param": params,
                       "Sol": sols,
                       "Obj Val": objvals,
                       "Mean Violation": mean_viols,
                       "Max Violation": max_viols,
                       "Num Violations": num_viols,
                       "Elapsed Time": elapseds})
    time.sleep(1)
    print(df.describe())
    print("Number of infeasible solutions: {}".format(np.sum(df["Num Violations"] > 0)))
    if penalty_growth:
        df.to_csv(f"result/rb_lrn{penalty_weight}_{num_blocks}-g.csv")
    else:
        df.to_csv(f"result/rb_lrn{penalty_weight}_{num_blocks}.csv")

def rndSte(loader_train, loader_test, loader_val, config, penalty_growth=False):
    print(config)
    # random seed
    np.random.seed(2)
    torch.manual_seed(2)
    torch.cuda.manual_seed(2)
    print(f"RS in RB for size {config.size}.")
    import neuromancer as nm
    from src.problem import nmRosenbrock
    from src.func.layer import netFC
    from src.func import roundSTEModel
    # config
    steepness = config.steepness
    num_blocks = config.size
    hlayers_sol = config.hlayers_sol
    hsize = config.hsize
    lr = config.lr
    penalty_weight = config.penalty
    project = config.project
    # init model
    from src.problem import msRosenbrock
    model = msRosenbrock(steepness, num_blocks, timelimit=1000)
    # build neural architecture for the solution map
    func = nm.modules.blocks.MLP(insize=num_blocks+1, outsize=2*num_blocks, bias=True,
                                 linear_map=nm.slim.maps["linear"],
                                 nonlin=nn.ReLU, hsizes=[hsize]*hlayers_sol)
    smap = nm.system.Node(func, ["p", "a"], ["x"], name="smap")
    # define rounding model
    rnd = roundSTEModel(param_keys=["p", "a"], var_keys=["x"], output_keys=["x_rnd"],
                        int_ind=model.int_ind, name="round")
    # build neuromancer problem for rounding
    components = nn.ModuleList([smap, rnd]).to("cuda")
    loss_fn = nmRosenbrock(["p", "a", "x_rnd"], steepness, num_blocks, penalty_weight)
    # train
    utils.train(components, loss_fn, loader_train, loader_val, lr, penalty_growth)
    # eval
    df = evaluate(components, loss_fn, model, loader_test, project)
    if penalty_growth:
        df.to_csv(f"result/rb_ste{penalty_weight}_{num_blocks}-g.csv")
    elif config.project:
        df.to_csv(f"result/rb_ste{penalty_weight}_{num_blocks}-p.csv")
    elif config.samples == 80000:
        df.to_csv(f"result/ste{penalty_weight}_{num_blocks}-l.csv")
    elif config.project:
        df.to_csv(f"result/rb_ste{penalty_weight}_{num_blocks}-p.csv")
    elif config.project and config.samples == 800:
        df.to_csv(f"result/rb_ste{penalty_weight}_{num_blocks}-s-p.csv")
    elif config.project and config.samples == 80000:
        df.to_csv(f"result/rb_ste{penalty_weight}_{num_blocks}-l-p.csv")
    else:
        df.to_csv(f"result/rb_ste{penalty_weight}_{num_blocks}.csv")


def evaluate(components, loss_fn, model, loader_test, project):
    # postprocessing
    if project:
        from src.postprocess.project import gradientProjection
        # project
        proj = gradientProjection([components[0]], [components[1]], loss_fn, "x")
    # init res
    params, sols, objvals, mean_viols, max_viols, num_viols, elapseds = [], [], [], [], [], [], []
    p_test = loader_test.dataset.datadict["p"]
    a_test = loader_test.dataset.datadict["a"]
    for p, a in tqdm(list(zip(p_test, a_test))):
        # data point as tensor
        datapoints = {"p": torch.tensor(np.array([p]), dtype=torch.float32).to("cuda"),
                      "a": torch.tensor(np.array([a]), dtype=torch.float32).to("cuda"),
                      "name": "test"}
        # infer
        components.eval()
        tick = time.time()
        with torch.no_grad():
            for comp in components:
                datapoints.update(comp(datapoints))
        if project:
            proj(datapoints)
        tock = time.time()
        # assign params
        model.set_param_val({"p":p, "a":a})
        # assign vars
        x = datapoints["x_rnd"]
        for i in range(len(model.vars["x"])):
            model.vars["x"][i].value = x[0,i].item()
        # get solutions
        xval, objval = model.get_val()
        params.append(list(p)+list(a))
        sols.append(list(list(xval.values())[0].values()))
        objvals.append(objval)
        viol = model.cal_violation()
        mean_viols.append(np.mean(viol))
        max_viols.append(np.max(viol))
        num_viols.append(np.sum(viol > 1e-6))
        elapseds.append(tock - tick)
    df = pd.DataFrame({"Param": params,
                       "Sol": sols,
                       "Obj Val": objvals,
                       "Mean Violation": mean_viols,
                       "Max Violation": max_viols,
                       "Num Violations": num_viols,
                       "Elapsed Time": elapseds})
    time.sleep(1)
    print(df.describe())
    print("Number of infeasible solutions: {}".format(np.sum(df["Num Violations"] > 0)))
    return df


def create_gurobi_model(test_case_params, steepness=50, num_blocks=1, timelimit=1000):
    # Extract parameters
    p = test_case_params['p']
    a = test_case_params['a']

    # Create Gurobi model
    model = gp.Model("rosenbrock")
    model.setParam('TimeLimit', timelimit)
    model.setParam('OutputFlag', 1)  # Suppress output by default

    # Create variables
    # x[2*i] are continuous variables
    # x[2*i+1] are integer variables
    x = {}
    for i in range(num_blocks):
        x[2*i] = model.addVar(vtype=gp.GRB.CONTINUOUS, name=f"x_{2*i}")
        x[2*i+1] = model.addVar(vtype=gp.GRB.INTEGER, name=f"x_{2*i+1}")

    # Set objective function: sum((a[i] - x[2*i])^2 + steepness * (x[2*i+1] - x[2*i]^2)^2)
    # Enable nonconvex quadratic optimization
    model.setParam('NonConvex', 2)

    obj = gp.QuadExpr()
    x_squared = {}  # Store auxiliary variables for x[2*i]^2

    for i in range(num_blocks):
        # First term: (a[i] - x[2*i])^2
        obj += (a[i] - x[2*i]) * (a[i] - x[2*i])

        # Second term: steepness * (x[2*i+1] - x[2*i]^2)^2
        # Create auxiliary variable for x[2*i]^2
        x_squared[i] = model.addVar(vtype=gp.GRB.CONTINUOUS, name=f"x_squared_{i}")

        # Add constraint: x_squared[i] = x[2*i]^2
        model.addConstr(x_squared[i] == x[2*i] * x[2*i], f"x_squared_constraint_{i}")

        # Add the Rosenbrock term: steepness * (x[2*i+1] - x_squared[i])^2
        obj += steepness * (x[2*i+1] - x_squared[i]) * (x[2*i+1] - x_squared[i])

    model.setObjective(obj, gp.GRB.MINIMIZE)

    # Add constraints
    # sum(x[2*i+1] for i in range(num_blocks)) >= num_blocks * p / 2
    model.addConstr(gp.quicksum(x[2*i+1] for i in range(num_blocks)) >= num_blocks * p / 2, "constraint1")

    # sum(x[2*i]^2 for i in range(num_blocks)) <= num_blocks * p
    model.addConstr(gp.quicksum(x[2*i] * x[2*i] for i in range(num_blocks)) <= num_blocks * p, "constraint2")

    # Linear constraints with random coefficients (matching original problem)
    rng = np.random.RandomState(17)
    b = rng.normal(scale=1, size=(num_blocks))
    q = rng.normal(scale=1, size=(num_blocks))

    # sum(b[i] * x[2*i] for i in range(num_blocks)) <= 0
    model.addConstr(gp.quicksum(b[i] * x[2*i] for i in range(num_blocks)) <= 0, "constraint3")

    # sum(q[i] * x[2*i+1] for i in range(num_blocks)) <= 0
    model.addConstr(gp.quicksum(q[i] * x[2*i+1] for i in range(num_blocks)) <= 0, "constraint4")

    # Store variables and parameters for later access
    model._variables = x
    model._x_squared = x_squared  # Store auxiliary variables
    model._params = test_case_params
    model._num_blocks = num_blocks

    return model


def extract_ml_solutions(result_dir="result", config_size=None):
    ml_solutions = {}
    # Filter CSV files based on config size if provided
    if config_size is not None:
        csv_files = [f for f in os.listdir(result_dir)
                    if f.startswith('rb_') and f.endswith('.csv') and f'{config_size}' in f]
    else:
        csv_files = [f for f in os.listdir(result_dir) if f.startswith('rb_') and f.endswith('.csv')]

    for csv_file in csv_files:
        method_name = csv_file.replace('.csv', '').replace('rb_', '')
        file_path = os.path.join(result_dir, csv_file)

        try:
            df = pd.read_csv(file_path)
            for idx, row in df.iterrows():
                if idx not in ml_solutions:
                    ml_solutions[idx] = {}

                # Parse parameter and solution strings
                param_str = row['Param']
                sol_str = row['Sol']

                # Convert string representations to lists
                params = ast.literal_eval(param_str)
                solution = ast.literal_eval(sol_str)

                ml_solutions[idx][method_name] = {
                    'params': params,
                    'solution': solution,
                    'obj_val': row['Obj Val'],
                    'elapsed_time': row['Elapsed Time']
                }
        except Exception as e:
            print(f"Warning: Could not process {csv_file}: {e}")

    return ml_solutions


def solve_with_warmstart(model, initial_solution, log_file_path=None):


    # Set initial solution values (warm start)
    variables = model._variables

    # Set start values for variables
    for i in range(min(len(initial_solution), len(variables))):
        if i in variables:
            variables[i].start = initial_solution[i]

    # Configure Gurobi logging
    if log_file_path:
        try:
            # Ensure log directory exists
            log_dir = os.path.dirname(log_file_path)
            if not os.path.exists(log_dir):
                os.makedirs(log_dir)

            # Set Gurobi logging parameters
            model.setParam('OutputFlag', 1)  # Enable output
            model.setParam('LogFile', log_file_path)  # Set log file path
        except Exception as e:
            print(f"Warning: Could not set up logging to {log_file_path}: {e}")

    # Capture Gurobi output to detect if warm start was used
    old_stdout = sys.stdout
    captured_output = io.StringIO()

    # Solve with warm start
    tick = time.time()
    try:
        # Temporarily redirect stdout to capture Gurobi messages
        sys.stdout = captured_output
        model.optimize()
        sys.stdout = old_stdout

        tock = time.time()
        solve_time = tock - tick

        # Analyze captured output to determine if warm start was used
        output_text = captured_output.getvalue()
        warmstart_used = "User MIP start produced solution" in output_text
        warmstart_rejected = "User MIP start did not produce" in output_text or "User MIP start violates" in output_text

        # Create warmstart info
        warmstart_info = {
            'initial_solution': initial_solution.copy(),
            'warmstart_used': warmstart_used,
            'warmstart_rejected': warmstart_rejected,
            'gurobi_output': output_text
        }

        # Check if solution was found or time limit reached
        if model.status == gp.GRB.OPTIMAL or model.status == gp.GRB.TIME_LIMIT:
            # Extract solution (if available)
            solution_dict = None
            objval = None

            if model.solCount > 0:  # At least one solution found
                solution_dict = {}
                for i, var in variables.items():
                    solution_dict[i] = var.x
                objval = model.objVal

            # Get bounds information
            best_bound = None
            best_objective = None

            try:
                if hasattr(model, 'objBound'):
                    best_bound = model.objBound  # Lower bound for minimization
                if model.solCount > 0:
                    best_objective = model.objVal  # Best incumbent solution
            except:
                pass

            # Get solver statistics
            solver_stats = {
                'status': model.status,
                'nodes_processed': model.nodeCount,
                'iterations': model.iterCount,
                'mip_gap': model.mipGap if hasattr(model, 'mipGap') else None,
                'best_bound': best_bound,
                'best_objective': best_objective,
                'solution_count': model.solCount
            }

            return solution_dict, objval, solve_time, solver_stats, warmstart_info
        else:
            # No solution found
            solver_stats = {
                'status': model.status,
                'nodes_processed': model.nodeCount if hasattr(model, 'nodeCount') else 0,
                'iterations': model.iterCount if hasattr(model, 'iterCount') else 0,
                'mip_gap': None,
                'best_bound': None,
                'best_objective': None,
                'solution_count': 0,
                'error': f'Solver terminated with status {model.status}'
            }
            return None, None, solve_time, solver_stats, warmstart_info

    except Exception as e:
        sys.stdout = old_stdout  # Restore stdout in case of error
        tock = time.time()
        solve_time = tock - tick
        print(f"Error in warm start solve: {e}")

        warmstart_info = {
            'initial_solution': initial_solution.copy(),
            'warmstart_used': False,
            'warmstart_rejected': False,
            'gurobi_output': f"Error: {str(e)}"
        }

        return None, None, solve_time, {'error': str(e)}, warmstart_info


def solve_direct_gurobi(model, log_file_path=None):


    # Clear any existing start values
    variables = model._variables
    for var in variables.values():
        var.start = gp.GRB.UNDEFINED

    # Configure Gurobi logging
    if log_file_path:
        try:
            # Ensure log directory exists
            log_dir = os.path.dirname(log_file_path)
            if not os.path.exists(log_dir):
                os.makedirs(log_dir)

            # Set Gurobi logging parameters
            model.setParam('OutputFlag', 1)  # Enable output
            model.setParam('LogFile', log_file_path)  # Set log file path
        except Exception as e:
            print(f"Warning: Could not set up logging to {log_file_path}: {e}")

    # Create empty warmstart info for consistency
    warmstart_info = {
        'initial_solution': None,
        'warmstart_used': False,
        'warmstart_rejected': False,
        'gurobi_output': 'Direct solve - no warm start'
    }

    # Solve directly
    tick = time.time()
    try:
        model.optimize()
        tock = time.time()
        solve_time = tock - tick

        # Check if solution was found or time limit reached
        if model.status == gp.GRB.OPTIMAL or model.status == gp.GRB.TIME_LIMIT:
            # Extract solution (if available)
            solution_dict = None
            objval = None

            if model.solCount > 0:  # At least one solution found
                solution_dict = {}
                for i, var in variables.items():
                    solution_dict[i] = var.x
                objval = model.objVal

            # Get bounds information
            best_bound = None
            best_objective = None

            try:
                if hasattr(model, 'objBound'):
                    best_bound = model.objBound  # Lower bound for minimization
                if model.solCount > 0:
                    best_objective = model.objVal  # Best incumbent solution
            except:
                pass

            # Get solver statistics
            solver_stats = {
                'status': model.status,
                'nodes_processed': model.nodeCount,
                'iterations': model.iterCount,
                'mip_gap': model.mipGap if hasattr(model, 'mipGap') else None,
                'best_bound': best_bound,
                'best_objective': best_objective,
                'solution_count': model.solCount
            }

            return solution_dict, objval, solve_time, solver_stats, warmstart_info
        else:
            # No solution found
            solver_stats = {
                'status': model.status,
                'nodes_processed': model.nodeCount if hasattr(model, 'nodeCount') else 0,
                'iterations': model.iterCount if hasattr(model, 'iterCount') else 0,
                'mip_gap': None,
                'best_bound': None,
                'best_objective': None,
                'solution_count': 0,
                'error': f'Solver terminated with status {model.status}'
            }
            return None, None, solve_time, solver_stats, warmstart_info

    except Exception as e:
        tock = time.time()
        solve_time = tock - tick
        print(f"Error in direct solve: {e}")
        return None, None, solve_time, {'error': str(e)}, warmstart_info


def warmstart(loader_test, config):
    print("="*80)
    print("PERFORMANCE COMPARISON: Direct Gurobi vs ML-Guided Warm Start")
    print("="*80)

    # Configuration
    steepness = config.steepness if hasattr(config, 'steepness') else 50
    num_blocks = config.size  # Use config.size for number of blocks
    timelimit = config.timelimit if hasattr(config, 'timelimit') else 60

    # Extract test case parameters
    p_test = loader_test.dataset.datadict["p"]
    a_test = loader_test.dataset.datadict["a"]
    test_cases = list(zip(p_test, a_test))

    # Extract ML solutions using config size
    print("Extracting ML solutions from CSV files...")
    ml_solutions = extract_ml_solutions(config_size=config.size)
    print(f"Found ML solutions for {len(ml_solutions)} test cases")

    # Results storage
    comparison_results = []

    # Create log directory
    import os
    log_dir = "gurobi_logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    print("\nRunning performance comparison...")
    print("-" * 80)

    # Process limited test cases for logging analysis
    max_test_cases = min(len(test_cases), 10)  # Limit to 10 test cases for logging analysis
    print(f"Processing {max_test_cases} test cases (limited for logging analysis)")

    for i in tqdm(range(max_test_cases), desc="Test Cases"):
        p, a = test_cases[i]
        test_params = {"p": p.item() if hasattr(p, 'item') else p,
                      "a": [a_val.item() if hasattr(a_val, 'item') else a_val for a_val in a]}

        print(f"\nTest Case {i+1}/{max_test_cases}")
        print(f"Parameters: p={test_params['p']:.3f}, a={test_params['a']}")

        # Create model for direct approach
        model_direct = create_gurobi_model(test_params, steepness, num_blocks, timelimit)

        # Approach ①: Direct Gurobi solving
        print("  ① Direct Gurobi solving...")
        direct_log_path = os.path.join(log_dir, f"rosenbrock_size{config.size}_instance{i}_Direct_Gurobi.log")
        direct_xval, direct_objval, direct_time, direct_stats, direct_warmstart_info = solve_direct_gurobi(model_direct, direct_log_path)

        # Approach ②: ML-guided warm start
        warmstart_results = {}
        if i in ml_solutions:
            for method_name, ml_data in ml_solutions[i].items():
                print(f"  ② Warm start with {method_name}...")

                # Create fresh model for warm start
                model_ws = create_gurobi_model(test_params, steepness, num_blocks, timelimit)
                ws_log_path = os.path.join(log_dir, f"rosenbrock_size{config.size}_instance{i}_{method_name}.log")
                ws_xval, ws_objval, ws_time, ws_stats, ws_warmstart_info = solve_with_warmstart(
                    model_ws, ml_data['solution'], ws_log_path
                )

                warmstart_results[method_name] = {
                    'xval': ws_xval,
                    'objval': ws_objval,
                    'solve_time': ws_time,
                    'stats': ws_stats,
                    'ml_objval': ml_data['obj_val'],
                    'ml_time': ml_data['elapsed_time'],
                    'warmstart_info': ws_warmstart_info
                }

        # Store results
        result_entry = {
            'test_case': i,
            'params': test_params,
            'direct': {
                'xval': direct_xval,
                'objval': direct_objval,
                'solve_time': direct_time,
                'stats': direct_stats
            },
            'warmstart': warmstart_results
        }
        comparison_results.append(result_entry)

        # Print summary for this test case
        direct_obj_str = f"{direct_objval:.4f}" if direct_objval is not None else "Failed"
        print(f"    Direct Gurobi: obj={direct_obj_str}, time={direct_time:.3f}s")

        for method_name, ws_result in warmstart_results.items():
            ws_obj = ws_result['objval']
            ws_time = ws_result['solve_time']
            ws_obj_str = f"{ws_obj:.4f}" if ws_obj is not None else "Failed"
            print(f"    Warm start ({method_name}): obj={ws_obj_str}, time={ws_time:.3f}s")

    # Export results to CSV
    export_results_to_csv(comparison_results, config)


def export_results_to_csv(comparison_results, config):
    print("\n EXPORTING RESULTS TO CSV")
    print("-" * 50)

    # Prepare data for CSV export
    csv_data = []

    for result in comparison_results:
        test_case_id = result['test_case']
        params = result['params']

        # Direct Gurobi results
        direct_result = result['direct']
        csv_data.append({
            'test_case_id': test_case_id,
            'method_name': 'Direct_Gurobi',
            'approach_type': 'Direct',
            'p_parameter': params['p'],
            'a_parameter': str(params['a']),
            'objective_value': direct_result['objval'] if direct_result['objval'] is not None else 'Failed',
            'solve_time_seconds': direct_result['solve_time'],
            'solver_status': direct_result['stats'].get('status', 'Unknown'),
            'nodes_processed': direct_result['stats'].get('nodes_processed', 'N/A'),
            'iterations': direct_result['stats'].get('iterations', 'N/A'),
            'mip_gap': direct_result['stats'].get('mip_gap', 'N/A'),
            'lower_bound': direct_result['stats'].get('best_bound', 'N/A'),
            'upper_bound': direct_result['stats'].get('best_objective', 'N/A'),
            'config_size': config.size,
            'success': 'Yes' if direct_result['objval'] is not None else 'No',
            'ml_objective_value': 'N/A',
            'ml_solve_time_seconds': 'N/A',
            'initial_solution': 'N/A',
            'warmstart_used': 'N/A'
        })

        # ML warm start results
        for method_name, ws_result in result['warmstart'].items():
            warmstart_info = ws_result.get('warmstart_info', {})
            initial_sol = warmstart_info.get('initial_solution', [])
            warmstart_used = warmstart_info.get('warmstart_used', False)

            # Format initial solution for display
            if initial_sol:
                initial_sol_str = '[' + ', '.join([f'{x:.4f}' for x in initial_sol]) + ']'
            else:
                initial_sol_str = 'N/A'

            # Determine warmstart usage status
            if warmstart_used:
                warmstart_status = 'Used'
            elif warmstart_info.get('warmstart_rejected', False):
                warmstart_status = 'Rejected'
            else:
                warmstart_status = 'Not_Used'

            csv_data.append({
                'test_case_id': test_case_id,
                'method_name': method_name,
                'approach_type': 'ML_Warmstart',
                'p_parameter': params['p'],
                'a_parameter': str(params['a']),
                'objective_value': ws_result['objval'] if ws_result['objval'] is not None else 'Failed',
                'solve_time_seconds': ws_result['solve_time'],
                'solver_status': ws_result['stats'].get('status', 'Unknown'),
                'nodes_processed': ws_result['stats'].get('nodes_processed', 'N/A'),
                'iterations': ws_result['stats'].get('iterations', 'N/A'),
                'mip_gap': ws_result['stats'].get('mip_gap', 'N/A'),
                'lower_bound': ws_result['stats'].get('best_bound', 'N/A'),
                'upper_bound': ws_result['stats'].get('best_objective', 'N/A'),
                'ml_objective_value': ws_result['ml_objval'],
                'ml_solve_time_seconds': ws_result['ml_time'],
                'config_size': config.size,
                'success': 'Yes' if ws_result['objval'] is not None else 'No',
                'initial_solution': initial_sol_str,
                'warmstart_used': warmstart_status
            })

    # Create DataFrame and export to CSV
    df = pd.DataFrame(csv_data)

    # Create results directory if it doesn't exist
    results_dir = "result"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    # Generate filename with timestamp and config info
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"performance_comparison_size{config.size}_{timestamp}.csv"
    filepath = os.path.join(results_dir, filename)

    # Export to CSV
    df.to_csv(filepath, index=False)