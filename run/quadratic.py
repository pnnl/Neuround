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
    if config.warmstart:
        model = msQuadratic(num_var, num_ineq, timelimit=10800)
        test_size = 10
    else:
        model = msQuadratic(num_var, num_ineq, timelimit=1000)
        test_size = 100
    # init df
    params, sols, objvals, mean_viols, max_viols, num_viols, elapseds = [], [], [], [], [], [], []
    logfile_suff = f"logs/cq_ex_{num_var}-{num_ineq}"
    infos_list = []
    # go through test data
    for idx, b in tqdm(enumerate(loader_test.dataset.datadict["b"][:test_size])):
        # log file path
        logfile = logfile_suff + f"_{idx}"
        # set params
        model.set_param_val({"b":b.cpu().numpy()})
        # solve
        tick = time.time()
        params.append(list(b.cpu().numpy()))
        try:
            if config.warmstart:
                xval, objval = model.solve(logfile=logfile+".log")
            else:
                xval, objval = model.solve()
            tock = time.time()
            # record info
            infos_list.append(model.last_solve_info)
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
    # save infos
    if config.warmstart:
        pd.DataFrame(infos_list).to_csv(logfile_suff+".csv", index=False)
    # save summary
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
    if config.warmstart:
        df.to_csv(f"result/cq_exact_{num_var}-{num_ineq}_3h.csv")
    else:
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
    if config.warmstart:
        logfile_suff = f"logs/cq_cls{penalty_weight}_{num_var}-{num_ineq}"
        df = evaluate_warmstart(components, loss_fn, model, loader_test, project, logfile_suff)
    else:
        df = evaluate(components, loss_fn, model, loader_test, project)
    # save
    if config.warmstart:
        df.to_csv(f"result/cq_cls{penalty_weight}_{num_var}-{num_ineq}-ws.csv")
    elif penalty_growth:
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
    if config.warmstart:
        logfile_suff = f"logs/cq_thd{penalty_weight}_{num_var}-{num_ineq}"
        df = evaluate_warmstart(components, loss_fn, model, loader_test, project, logfile_suff)
    else:
        df = evaluate(components, loss_fn, model, loader_test, project)
    # save
    if config.warmstart:
        df.to_csv(f"result/cq_thd{penalty_weight}_{num_var}-{num_ineq}-ws.csv")
    elif penalty_growth:
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


def evaluate_warmstart(components, loss_fn, model, loader_test, project, logfile_suff):
    # clone model
    model = model.clone()
    # set 3 hours time limit
    timelimit = 10800
    model.opt.options["timelimit"] = timelimit
    # postprocessing
    if project:
        from src.postprocess.project import gradientProjection
        # project
        proj = gradientProjection([components[0]], [components[1]], loss_fn, "x")
    # eval mode
    components.eval()
    # init res
    params, sols, objvals, mean_viols, max_viols, num_viols, elapseds = [], [], [], [], [], [], []
    infos_list = []
    for idx, b in tqdm(enumerate(loader_test.dataset.datadict["b"][:10])):
        # log file path
        logfile = logfile_suff + f"_{idx}"
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
        # assign params
        model.set_param_val({"b":b.cpu().numpy()})
        # get init solutions
        x_rnd = datapoints["x_rnd"][0].detach().cpu().numpy()
        init_sol = {"x": {i:v for i, v in enumerate(x_rnd)}}
        model.set_warm_start(init_sol)
        # solve
        xval, objval = model.solve(logfile=logfile+".log")
        tock = time.time()
        # record info
        infos_list.append(model.last_solve_info)
        # get solution
        params.append(list(b.cpu().numpy()))
        sols.append(list(list(xval.values())[0].values()))
        objvals.append(objval)
        viol = model.cal_violation()
        mean_viols.append(np.mean(viol))
        max_viols.append(np.max(viol))
        num_viols.append(np.sum(viol > 1e-6))
        elapseds.append(tock - tick)
    # save infos
    pd.DataFrame(infos_list).to_csv(logfile_suff+".csv", index=False)
    # # get summary
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
