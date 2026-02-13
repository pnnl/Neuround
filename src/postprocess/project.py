"""
Projection Gradient
"""

import numpy as np
import torch
from torch import nn
import neuromancer as nm

class gradientProjection(nn.Module):
    def __init__(self, pre_components, post_components, loss_fn, target_key, max_iters=1000, step_size=0.01, decay=1.0):
        super().__init__()
        self.pre_components = pre_components
        self.post_components = post_components
        self.loss_fn = loss_fn
        self.target_key = target_key
        self.max_iters = max_iters
        self.step_size = step_size
        self.decay = decay

    def forward(self, input_dict):
        # initialize decay multiplier
        d = 1.0
        # get target variables
        for comp in self.pre_components:
            input_dict.update(comp(input_dict))
        x = input_dict[self.target_key]
        # project gradient
        for _ in range(self.max_iters):
            # forward pass in components
            for comp in self.post_components:
                input_dict.update(comp(input_dict))
            # get corresponding violation
            viol = self.loss_fn.cal_constr_viol(input_dict)
            # check stopping condition
            if viol.max() < 1e-6:
                break
            # get gradients
            grad = torch.autograd.grad(viol.sum(), x)[0]
            # update
            x = x - d * self.step_size * grad
            d = self.decay * d
            # get data
            input_dict[self.target_key] = x
        return input_dict


if __name__ == "__main__":

    # random seed
    np.random.seed(42)
    torch.manual_seed(42)

    # init
    num_var = 100
    num_ineq = 100
    hlayers_sol = 5
    hlayers_rnd = 4
    hsize = 256
    batch_size = 64
    lr = 1e-3
    penalty_weight = 100
    num_data = 10000
    test_size = 1000
    val_size = 1000
    train_size = num_data - test_size - val_size

    # init mathematical model
    from src.problem import msQuadratic
    model = msQuadratic(num_var, num_ineq, timelimit=60)

    # data sample from uniform distribution
    b_samples = torch.from_numpy(np.random.uniform(-1, 1, size=(num_data, num_ineq))).float()
    data = {"b":b_samples}
    # data split
    from src.utils import data_split
    data_train, data_test, data_val = data_split(data, test_size=test_size, val_size=val_size)
    # torch dataloaders
    from torch.utils.data import DataLoader
    loader_train = DataLoader(data_train, batch_size, num_workers=0,
                              collate_fn=data_train.collate_fn, shuffle=True)
    loader_test  = DataLoader(data_test, batch_size, num_workers=0,
                              collate_fn=data_test.collate_fn, shuffle=False)
    loader_val   = DataLoader(data_val, batch_size, num_workers=0,
                              collate_fn=data_val.collate_fn, shuffle=False)

    # define neural architecture for the solution map smap(p) -> x
    import neuromancer as nm
    from src.func.layer import netFC
    func = netFC(input_dim=num_ineq, hidden_dims=[hsize]*hlayers_sol, output_dim=num_var)
    smap = nm.system.Node(func, ["b"], ["x"], name="smap")

    # define rounding model
    from src.func.layer import netFC
    from src.func import roundGumbelModel
    layers_rnd = netFC(input_dim=num_ineq+num_var, hidden_dims=[hsize]*hlayers_rnd,
                       output_dim=num_var)
    rnd = roundGumbelModel(layers=layers_rnd, param_keys=["b"], var_keys=["x"],
                           output_keys=["x_rnd"], int_ind=model.int_ind,
                           continuous_update=True, name="round")

    # build neuromancer components
    components = nn.ModuleList([smap, rnd])

    # build neuromancer problem
    from src.problem import nmQuadratic
    loss_fn = nmQuadratic(["b", "x_rnd"], num_var, num_ineq, penalty_weight)

    # training
    from src.problem.neuromancer.trainer import trainer
    epochs = 200                    # number of training epochs
    patience = 20                   # number of epochs with no improvement in eval metric to allow before early stopping
    warmup = 40                     # number of epochs to wait before enacting early stopping policies
    optimizer = torch.optim.AdamW(components.parameters(), lr=lr)
    # create a trainer for the problem
    my_trainer = trainer(components, loss_fn, optimizer, epochs=epochs,
                         patience=patience, warmup=warmup)
    # training for the rounding problem
    my_trainer.train(loader_train, loader_val)

    # project
    proj = gradientProjection([smap], [rnd], loss_fn, "x")

    # evaluate
    import time
    from tqdm import tqdm
    import pandas as pd
    params, sols, objvals, mean_viols, max_viols, num_viols, elapseds = [], [], [], [], [], [], []
    for b in tqdm(loader_test.dataset.datadict["b"][:100]):
        # data point as tensor
        datapoints = {"b": torch.unsqueeze(b, 0),
                      "name": "test"}
        # infer
        components.eval()
        tick = time.time()
        with torch.no_grad():
            for comp in components:
                datapoints.update(comp(datapoints))
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
