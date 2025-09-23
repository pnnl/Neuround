#!/usr/bin/env python
# coding: utf-8
"""
Experiment pipeline for IQP
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
import sys  
from io import StringIO


# turn off warning
import logging
logging.getLogger("pyomo.core").setLevel(logging.ERROR)

def exact(loader_test, config):
    print(config)
    # random seed
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    print(f"Ex in CQ for size {config.size}.")
    # config
    num_var = config.size
    num_ineq = config.size
    # init model
    from src.problem import msQuadratic
    model = msQuadratic(num_var, num_ineq, timelimit=1000)
    # init df
    params, sols, objvals, mean_viols, max_viols, num_viols, elapseds = [], [], [], [], [], [], []
    # go through test data
    for b in tqdm(loader_test.dataset.datadict["b"][:100]):
        # set params
        model.set_param_val({"b":b.cpu().numpy()})
        # solve
        tick = time.time()
        params.append(list(b.cpu().numpy()))
        try:
            xval, objval = model.solve()
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
    df.to_csv(f"result/cq_exact_{num_var}-{num_ineq}.csv")


def relRnd(loader_test, config):
    print(config)
    # random seed
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    print(f"RR in CQ for size {config.size}.")
    from src.heuristic import naive_round
    # config
    num_var = config.size
    num_ineq = config.size
    # init model
    from src.problem import msQuadratic
    model = msQuadratic(num_var, num_ineq, timelimit=1000)
    # init df
    params, sols, objvals, mean_viols, max_viols, num_viols, elapseds = [], [], [], [], [], [], []
    # go through test data
    for b in tqdm(loader_test.dataset.datadict["b"][:100]):
        # set params
        model.set_param_val({"b":b.cpu().numpy()})
        # relax
        model_rel = model.relax()
        # solve
        tick = time.time()
        params.append(list(b.cpu().numpy()))
        try:
            xval_rel, _ = model_rel.solve()
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
    df.to_csv(f"result/cq_rel_{num_var}-{num_ineq}.csv")


def root(loader_test, config):
    print(config)
    # random seed
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    print(f"N1 in CQ for size {config.size}.")
    # config
    num_var = config.size
    num_ineq = config.size
    # init model
    from src.problem import msQuadratic
    model = msQuadratic(num_var, num_ineq, timelimit=1000)
    model_heur = model.first_solution_heuristic(nodes_limit=1)
    # init df
    params, sols, objvals, mean_viols, max_viols, num_viols, elapseds = [], [], [], [], [], [], []
    # go through test data
    for b in tqdm(loader_test.dataset.datadict["b"][:100]):
        # set params
        model_heur.set_param_val({"b":b.cpu().numpy()})
        # solve
        tick = time.time()
        params.append(list(b.cpu().numpy()))
        try:
            xval, objval = model_heur.solve()
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
    df.to_csv(f"result/cq_root_{num_var}-{num_ineq}.csv")


def rndCls(loader_train, loader_test, loader_val, config, penalty_growth=False):
    print(config)
    # random seed
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    print(f"RC in CQ for size {config.size}.")
    import neuromancer as nm
    from src.problem import nmQuadratic
    from src.func.layer import netFC
    from src.func import roundGumbelModel
    # config
    num_var = config.size
    num_ineq = config.size
    hlayers_sol = config.hlayers_sol
    hlayers_rnd = config.hlayers_rnd
    hsize = config.hsize
    lr = config.lr
    penalty_weight = config.penalty
    project = config.project
    # init model
    from src.problem import msQuadratic
    model = msQuadratic(num_var, num_ineq, timelimit=1000)
    # build neural architecture for the solution map
    func = nm.modules.blocks.MLP(insize=num_ineq, outsize=num_var, bias=True,
                                 linear_map=nm.slim.maps["linear"],
                                 nonlin=nn.ReLU, hsizes=[hsize]*hlayers_sol)
    smap = nm.system.Node(func, ["b"], ["x"], name="smap")
    # define rounding model
    layers_rnd = netFC(input_dim=num_ineq+num_var,
                       hidden_dims=[hsize]*hlayers_rnd,
                       output_dim=num_var)
    rnd = roundGumbelModel(layers=layers_rnd, param_keys=["b"], var_keys=["x"],
                           output_keys=["x_rnd"], int_ind=model.int_ind,
                           continuous_update=True, name="round")
    # build neuromancer problem for rounding
    components = nn.ModuleList([smap, rnd]).to("cuda")
    loss_fn = nmQuadratic(["b", "x_rnd"], num_var, num_ineq, penalty_weight)
    # train
    utils.train(components, loss_fn, loader_train, loader_val, lr, penalty_growth)
    # eval
    df = evaluate(components, loss_fn, model, loader_test, project)
    if penalty_growth:
        df.to_csv(f"result/cq_cls{penalty_weight}_{num_var}-{num_ineq}-g.csv")
    elif config.samples == 800:
        df.to_csv(f"result/cq_cls{penalty_weight}_{num_var}-{num_ineq}-s.csv")
    elif config.samples == 80000:
        df.to_csv(f"result/cq_cls{penalty_weight}_{num_var}-{num_ineq}-l.csv")
    elif config.project:
        df.to_csv(f"result/cq_cls{penalty_weight}_{num_var}-{num_ineq}-p.csv")
    elif config.project and config.samples == 800:
        df.to_csv(f"result/cq_cls{penalty_weight}_{num_var}-{num_ineq}-s-p.csv")
    elif config.project and config.samples == 80000:
        df.to_csv(f"result/cq_cls{penalty_weight}_{num_var}-{num_ineq}-l-p.csv")
    else:
        df.to_csv(f"result/cq_cls{penalty_weight}_{num_var}-{num_ineq}.csv")


def rndThd(loader_train, loader_test, loader_val, config, penalty_growth=False):
    print(config)
    # random seed
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    print(f"LT in CQ for size {config.size}.")
    import neuromancer as nm
    from src.problem import nmQuadratic
    from src.func.layer import netFC
    from src.func import roundThresholdModel
    # config
    num_var = config.size
    num_ineq = config.size
    hlayers_sol = config.hlayers_sol
    hlayers_rnd = config.hlayers_rnd
    hsize = config.hsize
    lr = config.lr
    penalty_weight = config.penalty
    project = config.project
    # init model
    from src.problem import msQuadratic
    model = msQuadratic(num_var, num_ineq, timelimit=1000)
    # build neural architecture for the solution map
    func = nm.modules.blocks.MLP(insize=num_ineq, outsize=num_var, bias=True,
                                 linear_map=nm.slim.maps["linear"],
                                 nonlin=nn.ReLU, hsizes=[hsize]*hlayers_sol)
    smap = nm.system.Node(func, ["b"], ["x"], name="smap")
    # define rounding model
    layers_rnd = netFC(input_dim=num_ineq+num_var,
                       hidden_dims=[hsize]*hlayers_rnd,
                       output_dim=num_var)
    rnd = roundThresholdModel(layers=layers_rnd, param_keys=["b"], var_keys=["x"],
                              output_keys=["x_rnd"], int_ind=model.int_ind,
                              continuous_update=True, name="round")
    # build neuromancer problem for rounding
    components = nn.ModuleList([smap, rnd]).to("cuda")
    loss_fn = nmQuadratic(["b", "x_rnd"], num_var, num_ineq, penalty_weight)
    # train
    utils.train(components, loss_fn, loader_train, loader_val, lr, penalty_growth)
    # eval
    df = evaluate(components, loss_fn, model, loader_test, project)
    if penalty_growth:
        df.to_csv(f"result/cq_thd{penalty_weight}_{num_var}-{num_ineq}-g.csv")
    elif config.samples == 800:
        df.to_csv(f"result/cq_thd{penalty_weight}_{num_var}-{num_ineq}-s.csv")
    elif config.samples == 80000:
        df.to_csv(f"result/cq_thd{penalty_weight}_{num_var}-{num_ineq}-l.csv")
    elif config.project:
        df.to_csv(f"result/cq_thd{penalty_weight}_{num_var}-{num_ineq}-p.csv")
    elif config.project and config.samples == 800:
        df.to_csv(f"result/cq_thd{penalty_weight}_{num_var}-{num_ineq}-s-p.csv")
    elif config.project and config.samples == 80000:
        df.to_csv(f"result/cq_thd{penalty_weight}_{num_var}-{num_ineq}-l-p.csv")
    else:
        df.to_csv(f"result/cq_thd{penalty_weight}_{num_var}-{num_ineq}.csv")


def lrnRnd(loader_train, loader_test, loader_val, config, penalty_growth=False):
    print(config)
    # random seed
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    print(f"RL in CQ for size {config.size}.")
    import neuromancer as nm
    from src.problem import nmQuadratic
    from src.func.layer import netFC
    from src.func import roundThresholdModel
    # config
    num_var = config.size
    num_ineq = config.size
    hlayers_sol = config.hlayers_sol
    hsize = config.hsize
    lr = config.lr
    penalty_weight = config.penalty
    # init model
    from src.problem import msQuadratic
    model = msQuadratic(num_var, num_ineq, timelimit=1000)
    # build neural architecture for the solution map
    func = nm.modules.blocks.MLP(insize=num_ineq, outsize=num_var, bias=True,
                                 linear_map=nm.slim.maps["linear"],
                                 nonlin=nn.ReLU, hsizes=[hsize]*hlayers_sol)
    smap = nm.system.Node(func, ["b"], ["x"], name="smap")
    # build neuromancer problem for rounding
    components = nn.ModuleList([smap]).to("cuda")
    loss_fn = nmQuadratic(["b", "x"], num_var, num_ineq, penalty_weight)
    # train
    utils.train(components, loss_fn, loader_train, loader_val, lr, penalty_growth)
    # eval
    from src.heuristic import naive_round
    params, sols, objvals, mean_viols, max_viols, num_viols, elapseds = [], [], [], [], [], [], []
    for b in tqdm(loader_test.dataset.datadict["b"][:100]):
        # data point as tensor
        datapoints = {"b": torch.unsqueeze(b, 0).to("cuda"),
                      "name": "test"}
        # infer
        components.eval()
        tick = time.time()
        with torch.no_grad():
            for comp in components:
                datapoints.update(comp(datapoints))
        tock = time.time()
        # assign params
        model.set_param_val({"b":b.cpu().numpy()})
        # assign vars
        x = datapoints["x"]
        for i in range(num_var):
            model.vars["x"][i].value = x[0,i].item()
        # get solutions
        xval_rel, _ = model.get_val()
        xval, objval = naive_round(xval_rel, model)
        params.append(list(b.cpu().numpy()))
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
        df.to_csv(f"result/cq_lrn{penalty_weight}_{num_var}-{num_ineq}-g.csv")
    elif config.samples == 800:
        df.to_csv(f"result/cq_lrn{penalty_weight}_{num_var}-{num_ineq}-s.csv")
    elif config.samples == 80000:
        df.to_csv(f"result/cq_lrn{penalty_weight}_{num_var}-{num_ineq}-l.csv")
    else:
        df.to_csv(f"result/cq_lrn{penalty_weight}_{num_var}-{num_ineq}.csv")


def rndSte(loader_train, loader_test, loader_val, config, penalty_growth=False):
    print(config)
    # random seed
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    print(f"RS in CQ for size {config.size}.")
    import neuromancer as nm
    from src.problem import nmQuadratic
    from src.func.layer import netFC
    from src.func import roundSTEModel
    # config
    num_var = config.size
    num_ineq = config.size
    hlayers_sol = config.hlayers_sol
    hsize = config.hsize
    lr = config.lr
    penalty_weight = config.penalty
    project = config.project
    # init model
    from src.problem import msQuadratic
    model = msQuadratic(num_var, num_ineq, timelimit=1000)
    # build neural architecture for the solution map
    func = nm.modules.blocks.MLP(insize=num_ineq, outsize=num_var, bias=True,
                                 linear_map=nm.slim.maps["linear"],
                                 nonlin=nn.ReLU, hsizes=[hsize]*hlayers_sol)
    smap = nm.system.Node(func, ["b"], ["x"], name="smap")
    # define rounding model
    rnd = roundSTEModel(param_keys=["b"], var_keys=["x"],  output_keys=["x_rnd"], int_ind=model.int_ind, name="round")
    # build neuromancer problem for rounding
    components = nn.ModuleList([smap, rnd]).to("cuda")
    loss_fn = nmQuadratic(["b", "x_rnd"], num_var, num_ineq, penalty_weight)
    # train
    utils.train(components, loss_fn, loader_train, loader_val, lr, penalty_growth)
    # eval
    df = evaluate(components, loss_fn, model, loader_test, project)
    if penalty_growth:
        df.to_csv(f"result/cq_ste{penalty_weight}_{num_var}-{num_ineq}-g.csv")
    elif config.samples == 800:
        df.to_csv(f"result/cq_ste{penalty_weight}_{num_var}-{num_ineq}-s.csv")
    elif config.samples == 80000:
        df.to_csv(f"result/cq_ste{penalty_weight}_{num_var}-{num_ineq}-l.csv")
    elif config.project:
        df.to_csv(f"result/cq_ste{penalty_weight}_{num_var}-{num_ineq}-p.csv")
    elif config.project and config.samples == 800:
        df.to_csv(f"result/cq_ste{penalty_weight}_{num_var}-{num_ineq}-s-p.csv")
    elif config.project and config.samples == 80000:
        df.to_csv(f"result/cq_ste{penalty_weight}_{num_var}-{num_ineq}-l-p.csv")
    else:
        df.to_csv(f"result/cq_ste{penalty_weight}_{num_var}-{num_ineq}.csv")


def evaluate(components, loss_fn, model, loader_test, project):
    # postprocessing
    if project:
        from src.postprocess.project import gradientProjection
        # project
        proj = gradientProjection([components[0]], [components[1]], loss_fn, "x")
    # eval
    components.eval()
    # init res
    params, sols, objvals, mean_viols, max_viols, num_viols, elapseds = [], [], [], [], [], [], []
    for b in tqdm(loader_test.dataset.datadict["b"][:100]):
        # data point as tensor
        datapoints = {"b": torch.unsqueeze(b, 0).to("cuda"),
                      "name": "test"}
        # infer
        tick = time.time()
        with torch.no_grad():
            for comp in components:
                datapoints.update(comp(datapoints))
        if project:
            proj(datapoints)
        tock = time.time()
        # assign params
        model.set_param_val({"b":b.cpu().numpy()})
        # assign vars
        x = datapoints["x_rnd"]
        for i in range(len(model.vars["x"])):
            model.vars["x"][i].value = x[0,i].item()
        # get solutions
        xval, objval = model.get_val()
        params.append(list(b.cpu().numpy()))
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
    print(df.describe())
    print("Number of infeasible solutions: {}".format(np.sum(df["Num Violations"] > 0)))
    return df


def create_gurobi_model(test_case_params, num_var, num_ineq, timelimit=60):
    # Extract parameters
    b = test_case_params['b']

    # Create Gurobi model
    model = gp.Model("quadratic")
    model.setParam('TimeLimit', timelimit)
    model.setParam('OutputFlag', 1)  # Enable output by default

    # Fixed coefficients (same as in the problem definition)
    rng = np.random.RandomState(17)
    Q = 0.01 * np.diag(rng.random(size=num_var))
    p = 0.1 * rng.random(num_var)
    A = rng.normal(scale=0.1, size=(num_ineq, num_var))

    # Create variables (all integer variables)
    x = {}
    for j in range(num_var):
        x[j] = model.addVar(vtype=gp.GRB.INTEGER, name=f"x_{j}")

    # Set objective function: 1/2 x^T Q x + p^T x
    obj = gp.QuadExpr()
    for j in range(num_var):
        # Quadratic term: 1/2 * Q[j,j] * x[j]^2
        obj += 0.5 * Q[j,j] * x[j] * x[j]
        # Linear term: p[j] * x[j]
        obj += p[j] * x[j]

    model.setObjective(obj, gp.GRB.MINIMIZE)

    # Add constraints: A x <= b
    for i in range(num_ineq):
        model.addConstr(gp.quicksum(A[i,j] * x[j] for j in range(num_var)) <= b[i], f"constraint_{i}")

    # Store variables and parameters for later access
    model._variables = x
    model._params = test_case_params
    model._num_var = num_var
    model._num_ineq = num_ineq
    model._Q = Q
    model._p = p
    model._A = A

    return model


def extract_ml_solutions(result_dir="result", config_size=None):
    ml_solutions = {}

    # Check if result directory exists
    if not os.path.exists(result_dir):
        print(f"Result directory '{result_dir}' does not exist. No ML solutions available.")
        return ml_solutions

    # Filter CSV files based on config size if provided
    if config_size is not None:
        csv_files = [f for f in os.listdir(result_dir)
                    if f.startswith('cq_') and f.endswith('.csv') and f'{config_size}' in f]
    else:
        csv_files = [f for f in os.listdir(result_dir) if f.startswith('cq_') and f.endswith('.csv')]

    print(f"Found CSV files: {csv_files}")

    for csv_file in csv_files:
        try:
            df = pd.read_csv(os.path.join(result_dir, csv_file))

            # Extract method name from filename
            method_name = csv_file.replace('.csv', '').replace('cq_', '')

            # Process each row in the CSV
            for idx, row in df.iterrows():
                if idx >= 100:  # Limit to first 100 test cases
                    break

                # Parse solution from string representation
                sol_str = row['Sol']
                if pd.isna(sol_str) or sol_str == 'None':
                    continue

                try:
                    # Convert string representation to list
                    if isinstance(sol_str, str):
                        # Remove brackets and split by comma
                        sol_str = sol_str.strip('[]')
                        solution = [float(x.strip()) for x in sol_str.split(',')]
                    else:
                        continue

                    # Store solution data
                    if idx not in ml_solutions:
                        ml_solutions[idx] = {}

                    ml_solutions[idx][method_name] = {
                        'solution': solution,
                        'obj_val': row['Obj Val'],
                        'elapsed_time': row['Elapsed Time']
                    }
                except Exception as e:
                    print(f"Error parsing solution for {method_name}, row {idx}: {e}")
                    continue

        except Exception as e:
            print(f"Error reading {csv_file}: {e}")
            continue

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

    # Capture stdout to analyze warm start usage
    old_stdout = sys.stdout
    captured_output = StringIO()

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

            # Calculate MIP gap
            mip_gap = None
            if best_bound is not None and best_objective is not None and abs(best_objective) > 1e-10:
                mip_gap = abs(best_objective - best_bound) / abs(best_objective)

            # Collect solver statistics
            solver_stats = {
                'status': 'Optimal' if model.status == gp.GRB.OPTIMAL else 'Time_Limit',
                'nodes_processed': getattr(model, 'nodeCount', 'N/A'),
                'iterations': getattr(model, 'iterCount', 'N/A'),
                'mip_gap': mip_gap,
                'best_bound': best_bound,
                'best_objective': best_objective
            }

            return solution_dict, objval, solve_time, solver_stats, warmstart_info
        else:
            # No solution found
            solver_stats = {
                'status': f'Status_{model.status}',
                'nodes_processed': getattr(model, 'nodeCount', 'N/A'),
                'iterations': getattr(model, 'iterCount', 'N/A'),
                'mip_gap': 'N/A',
                'best_bound': getattr(model, 'objBound', 'N/A'),
                'best_objective': 'N/A'
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

            # Calculate MIP gap
            mip_gap = None
            if best_bound is not None and best_objective is not None and abs(best_objective) > 1e-10:
                mip_gap = abs(best_objective - best_bound) / abs(best_objective)

            # Collect solver statistics
            solver_stats = {
                'status': 'Optimal' if model.status == gp.GRB.OPTIMAL else 'Time_Limit',
                'nodes_processed': getattr(model, 'nodeCount', 'N/A'),
                'iterations': getattr(model, 'iterCount', 'N/A'),
                'mip_gap': mip_gap,
                'best_bound': best_bound,
                'best_objective': best_objective
            }

            return solution_dict, objval, solve_time, solver_stats, warmstart_info
        else:
            # No solution found
            solver_stats = {
                'status': f'Status_{model.status}',
                'nodes_processed': getattr(model, 'nodeCount', 'N/A'),
                'iterations': getattr(model, 'iterCount', 'N/A'),
                'mip_gap': 'N/A',
                'best_bound': getattr(model, 'objBound', 'N/A'),
                'best_objective': 'N/A'
            }

            return None, None, solve_time, solver_stats, warmstart_info

    except Exception as e:
        tock = time.time()
        solve_time = tock - tick
        print(f"Error in direct solve: {e}")
        return None, None, solve_time, {'error': str(e)}, warmstart_info


def warmstart(loader_test, config):
    print("="*80)
    print("PERFORMANCE COMPARISON: Direct Gurobi vs ML-Guided Warm Start (Quadratic)")
    print("="*80)

    # Configuration
    num_var = config.size
    num_ineq = config.size
    timelimit = config.timelimit if hasattr(config, 'timelimit') else 60

    # Extract test case parameters
    b_test = loader_test.dataset.datadict["b"]
    test_cases = list(b_test)

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
        b = test_cases[i]
        test_params = {"b": [b_val.item() if hasattr(b_val, 'item') else b_val for b_val in b]}

        print(f"\nTest Case {i+1}/{max_test_cases}")
        print(f"Parameters: b={test_params['b']}")

        # Create model for direct approach
        model_direct = create_gurobi_model(test_params, num_var, num_ineq, timelimit)

        # Approach ①: Direct Gurobi solving
        print("  ① Direct Gurobi solving...")
        direct_log_path = os.path.join(log_dir, f"quadratic_size{config.size}_instance{i}_Direct_Gurobi.log")
        direct_xval, direct_objval, direct_time, direct_stats, direct_warmstart_info = solve_direct_gurobi(model_direct, direct_log_path)

        # Approach ②: ML-guided warm start
        warmstart_results = {}
        if i in ml_solutions:
            for method_name, ml_data in ml_solutions[i].items():
                print(f"  ② Warm start with {method_name}...")

                # Create fresh model for warm start
                model_ws = create_gurobi_model(test_params, num_var, num_ineq, timelimit)
                ws_log_path = os.path.join(log_dir, f"quadratic_size{config.size}_instance{i}_{method_name}.log")
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
    import os
    import datetime

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
            'b_parameter': str(params['b']),
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
                'b_parameter': str(params['b']),
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
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"performance_comparison_quadratic_size{config.size}_{timestamp}.csv"
    filepath = os.path.join(results_dir, filename)

    # Export to CSV
    df.to_csv(filepath, index=False)

    print(f"Results exported to: {filepath}")
    print(f"Total records: {len(csv_data)}")