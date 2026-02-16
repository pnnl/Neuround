#!/usr/bin/env python
# coding: utf-8
"""
Experiment pipeline for INC
"""
import random
import time
import numpy as np
import pandas as pd
import torch
from torch import nn
from tqdm import tqdm
from run import utils

# turn off warning
import logging
logging.getLogger("pyomo.core").setLevel(logging.ERROR)

def exact(loader_test, config):
    print(config)
    # random seed
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    print(f"EX in NC for size {config.size}.")
    # config
    num_var = config.size
    num_ineq = config.size
    # init model
    from src.problem import msNonconvex
    model = msNonconvex(num_var, num_ineq, timelimit=1000)
    # init df
    params, sols, objvals, mean_viols, max_viols, num_viols, elapseds = [], [], [], [], [], [], []
    # go through test data
    b_test = loader_test.dataset.datadict["b"][:100]
    d_test = loader_test.dataset.datadict["d"][:100]
    for b, d in tqdm(list(zip(b_test, d_test))):
        # set params
        model.set_param_val({"b":b.cpu().numpy(), "d":d.cpu().numpy()})
        # solve
        tick = time.time()
        params.append(list(b.cpu().numpy())+list(d.cpu().numpy()))
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
    df.to_csv(f"result/nc_exact_{num_var}-{num_ineq}_new.csv")


def relRnd(loader_test, config):
    print(config)
    # random seed
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    print(f"RR in NC for size {config.size}.")
    from src.heuristic import naive_round
    # config
    num_var = config.size
    num_ineq = config.size
    # init model
    from src.problem import msNonconvex
    model = msNonconvex(num_var, num_ineq, timelimit=1000)
    # init df
    params, sols, objvals, mean_viols, max_viols, num_viols, elapseds = [], [], [], [], [], [], []
    # go through test data
    b_test = loader_test.dataset.datadict["b"][:100]
    d_test = loader_test.dataset.datadict["d"][:100]
    for b, d in tqdm(list(zip(b_test, d_test))):
        # set params
        model.set_param_val({"b":b.cpu().numpy()})
        # relax
        model_rel = model.relax()
        # solve
        tick = time.time()
        params.append(list(b.cpu().numpy())+list(d.cpu().numpy()))
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
    df.to_csv(f"result/nc_rel_{num_var}-{num_ineq}_new.csv")


def root(loader_test, config):
    print(config)
    # random seed
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    print(f"N1 in NC for size {config.size}.")
    # config
    num_var = config.size
    num_ineq = config.size
    # init model
    from src.problem import msNonconvex
    model = msNonconvex(num_var, num_ineq, timelimit=1000)
    model_heur = model.first_solution_heuristic(nodes_limit=1)
    # init df
    params, sols, objvals, mean_viols, max_viols, num_viols, elapseds = [], [], [], [], [], [], []
    # go through test data
    b_test = loader_test.dataset.datadict["b"][:100]
    d_test = loader_test.dataset.datadict["d"][:100]
    for b, d in tqdm(list(zip(b_test, d_test))):
        # set params
        model_heur.set_param_val({"b":b.cpu().numpy(), "d":d.cpu().numpy()})
        # solve
        tick = time.time()
        params.append(list(b.cpu().numpy())+list(d.cpu().numpy()))
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
    df.to_csv(f"result/nc_root_{num_var}-{num_ineq}_new.csv")


def rndCls(loader_train, loader_test, loader_val, config, penalty_growth=False):
    print(config)
    # random seed
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    print(f"RC in NC for size {config.size}.")
    import neuromancer as nm
    from src.problem import nmNonconvex
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
    from src.problem import msNonconvex
    model = msNonconvex(num_var, num_ineq, timelimit=1000)
    # build neural architecture for the solution map
    func = nm.modules.blocks.MLP(insize=num_ineq*2, outsize=num_var, bias=True,
                                 linear_map=nm.slim.maps["linear"],
                                 nonlin=nn.ReLU, hsizes=[hsize]*hlayers_sol)
    smap = nm.system.Node(func, ["b", "d"], ["x"], name="smap")
    # define rounding model
    layers_rnd = netFC(input_dim=num_ineq*2+num_var,
                       hidden_dims=[hsize]*hlayers_rnd,
                       output_dim=num_var)
    rnd = roundGumbelModel(layers=layers_rnd, param_keys=["b", "d"], var_keys=["x"],
                           output_keys=["x_rnd"], int_ind=model.int_ind,
                           continuous_update=True, name="round")
    # build neuromancer problem for rounding
    components = nn.ModuleList([smap, rnd]).to("cuda")
    loss_fn = nmNonconvex(["b", "d", "x_rnd"], num_var, num_ineq, penalty_weight)
    # train
    utils.train(components, loss_fn, loader_train, loader_val, lr, penalty_growth)
    # eval
    df = evaluate(components, loss_fn, model, loader_test, project)
    if penalty_growth:
        df.to_csv(f"result/nc_cls{penalty_weight}_{num_var}-{num_ineq}-g_new.csv")
    elif config.project:
        df.to_csv(f"result/nc_cls{penalty_weight}_{num_var}-{num_ineq}-p_new.csv")
    else:
        df.to_csv(f"result/nc_cls{penalty_weight}_{num_var}-{num_ineq}_new.csv")


def rndThd(loader_train, loader_test, loader_val, config, penalty_growth=False):
    print(config)
    # random seed
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    print(f"LT in NC for size {config.size}.")
    import neuromancer as nm
    from src.problem import nmNonconvex
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
    from src.problem import msNonconvex
    model = msNonconvex(num_var, num_ineq, timelimit=1000)
    # build neural architecture for the solution map
    func = nm.modules.blocks.MLP(insize=num_ineq*2, outsize=num_var, bias=True,
                                 linear_map=nm.slim.maps["linear"],
                                 nonlin=nn.ReLU, hsizes=[hsize]*hlayers_sol)
    smap = nm.system.Node(func, ["b", "d"], ["x"], name="smap")
    # define rounding model
    layers_rnd = netFC(input_dim=num_ineq*2+num_var,
                       hidden_dims=[hsize]*hlayers_rnd,
                       output_dim=num_var)
    rnd = roundThresholdModel(layers=layers_rnd, param_keys=["b", "d"], var_keys=["x"],
                              output_keys=["x_rnd"], int_ind=model.int_ind,
                              continuous_update=True, name="round")
    # build neuromancer problem for rounding
    components = nn.ModuleList([smap, rnd]).to("cuda")
    loss_fn = nmNonconvex(["b", "d", "x_rnd"], num_var, num_ineq, penalty_weight)
    # train
    utils.train(components, loss_fn, loader_train, loader_val, lr, penalty_growth)
    # eval
    df = evaluate(components, loss_fn, model, loader_test, project)
    if penalty_growth:
        df.to_csv(f"result/nc_thd{penalty_weight}_{num_var}-{num_ineq}-g_new.csv")
    elif config.project:
        df.to_csv(f"result/nc_thd{penalty_weight}_{num_var}-{num_ineq}-p_new.csv")
    else:
        df.to_csv(f"result/nc_thd{penalty_weight}_{num_var}-{num_ineq}_new.csv")


def lrnRnd(loader_train, loader_test, loader_val, config, penalty_growth=False):
    print(config)
    # random seed
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    print(f"RL in NC for size {config.size}.")
    import neuromancer as nm
    from src.problem import nmNonconvex
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
    from src.problem import msNonconvex
    model = msNonconvex(num_var, num_ineq, timelimit=1000)
    # build neural architecture for the solution map
    func = nm.modules.blocks.MLP(insize=num_ineq*2, outsize=num_var, bias=True,
                                 linear_map=nm.slim.maps["linear"],
                                 nonlin=nn.ReLU, hsizes=[hsize]*hlayers_sol)
    smap = nm.system.Node(func, ["b", "d"], ["x"], name="smap")
    # build neuromancer problem for rounding
    components = nn.ModuleList([smap]).to("cuda")
    loss_fn = nmNonconvex(["b", "d", "x"], num_var, num_ineq, penalty_weight)
    # train
    utils.train(components, loss_fn, loader_train, loader_val, lr, penalty_growth)
    # eval
    from src.heuristic import naive_round
    params, sols, objvals, mean_viols, max_viols, num_viols, elapseds = [], [], [], [], [], [], []
    b_test = loader_test.dataset.datadict["b"][:100]
    d_test = loader_test.dataset.datadict["d"][:100]
    for b, d in tqdm(list(zip(b_test, d_test))):
        # data point as tensor
        datapoints = {"b": torch.unsqueeze(b, 0).to("cuda"),
                      "d": torch.unsqueeze(d, 0).to("cuda"),
                      "name": "test"}
        # infer
        components.eval()
        tick = time.time()
        with torch.no_grad():
            for comp in components:
                datapoints.update(comp(datapoints))
        tock = time.time()
        # assign params
        model.set_param_val({"b":b.cpu().numpy(), "d":d.cpu().numpy()})
        # assign vars
        x = datapoints["x"]
        for i in range(num_var):
            model.vars["x"][i].value = x[0,i].item()
        # get solutions
        xval_rel, _ = model.get_val()
        xval, objval = naive_round(xval_rel, model)
        params.append(list(b.cpu().numpy())+list(d.cpu().numpy()))
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
        df.to_csv(f"result/nc_lrn{penalty_weight}_{num_var}-{num_ineq}-g_new.csv")
    else:
        df.to_csv(f"result/nc_lrn{penalty_weight}_{num_var}-{num_ineq}_new.csv")


def rndSte(loader_train, loader_test, loader_val, config, penalty_growth=False):
    print(config)
    # random seed
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    print(f"RS in NC for size {config.size}.")
    import neuromancer as nm
    from src.problem import nmNonconvex
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
    from src.problem import msNonconvex
    model = msNonconvex(num_var, num_ineq, timelimit=1000)
    # build neural architecture for the solution map
    func = nm.modules.blocks.MLP(insize=num_ineq*2, outsize=num_var, bias=True,
                                 linear_map=nm.slim.maps["linear"],
                                 nonlin=nn.ReLU, hsizes=[hsize]*hlayers_sol)
    smap = nm.system.Node(func, ["b", "d"], ["x"], name="smap")
    # define rounding model
    rnd = roundSTEModel(param_keys=["b", "d"], var_keys=["x"], output_keys=["x_rnd"], int_ind=model.int_ind, name="round")
    # build neuromancer problem for rounding
    components = nn.ModuleList([smap, rnd]).to("cuda")
    loss_fn = nmNonconvex(["b", "d", "x_rnd"], num_var, num_ineq, penalty_weight)
    # train
    utils.train(components, loss_fn, loader_train, loader_val, lr, penalty_growth)
    # eval
    df = evaluate(components, loss_fn, model, loader_test, project)
    if penalty_growth:
        df.to_csv(f"result/nc_ste{penalty_weight}_{num_var}-{num_ineq}-g_new.csv")
    elif config.project:
        df.to_csv(f"result/nc_ste{penalty_weight}_{num_var}-{num_ineq}-p_new.csv")
    else:
        df.to_csv(f"result/nc_ste{penalty_weight}_{num_var}-{num_ineq}_new.csv")


def evaluate(components, loss_fn, model, loader_test, project):
    # postprocessing
    if project:
        from src.postprocess.project import gradientProjection
        # project
        proj = gradientProjection([components[0]], [components[1]], loss_fn, "x")
    # eval model
    components.eval()
    # init res
    params, sols, objvals, mean_viols, max_viols, num_viols, elapseds = [], [], [], [], [], [], []
    b_test = loader_test.dataset.datadict["b"][:100]
    d_test = loader_test.dataset.datadict["d"][:100]
    for b, d in tqdm(list(zip(b_test, d_test))):
        # data point as tensor
        datapoints = {"b": torch.unsqueeze(b, 0).to("cuda"),
                      "d": torch.unsqueeze(d, 0).to("cuda"),
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
        model.set_param_val({"b":b.cpu().numpy(), "d":d.cpu().numpy()})
        # assign vars
        x = datapoints["x_rnd"]
        for i in range(len(model.vars["x"])):
            model.vars["x"][i].value = x[0,i].item()
        # get solutions
        xval, objval = model.get_val()
        params.append(list(b.cpu().numpy())+list(d.cpu().numpy()))
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
