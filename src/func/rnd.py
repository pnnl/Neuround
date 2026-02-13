"""
Learning to round
"""
from collections import defaultdict

import torch
from torch import nn

from src.func.ste import diffFloor, diffBinarize, diffGumbelBinarize, thresholdBinarize

class roundSTEModel(nn.Module):
    """
    STE model to round integer variables
    """
    def __init__(self, param_keys, var_keys, output_keys=[],
                 int_ind=defaultdict(list), bin_ind=defaultdict(list),
                 tolerance=1e-3, temperature=1.0, name="Rounding"):
        super(roundSTEModel, self).__init__()
        # data keys
        self.param_keys, self.var_keys = param_keys, var_keys
        self.input_keys = self.param_keys + self.var_keys
        self.output_keys = output_keys if output_keys else self.var_keys
        # index of integer and binary variables
        self.int_ind = int_ind
        self.bin_ind = bin_ind
        # random temperature
        self.temperature = temperature
        # binarize
        self.floor, self.bin = diffFloor(), diffGumbelBinarize(temperature=self.temperature)
        # name
        self.name = name

    def forward(self, data):
        # rounding
        output_data = self._process_rounding(data)
        return output_data

    def _process_rounding(self, data):
        output_data = {}
        for k_in, k_out in zip(self.var_keys, self.output_keys):
            # get rounding
            x_rnd = self._round_vars(data, k_in)
            output_data[k_out] = x_rnd
        return output_data

    def _round_vars(self, data, key):
        # get index
        int_ind = self.int_ind[key]
        bin_ind = self.bin_ind[key]
        # load x
        x = data[key].clone()
        ###################### integer ######################
        # floor(x)
        x_flr = self.floor(x[:,int_ind])
        # bin(h): binary 0 for floor, 1 for ceil
        bnr = self.bin(x[:,int_ind] - x_flr.detach() - 0.5)
        # update rounding for integer variables int(x) = floor(x) + bin(h)
        x[:, int_ind] = x_flr + bnr
        ###################### binary ######################
        # update rounding for binary variables: bin(x) = bin(h)
        x[:, bin_ind] = self.bin(x[:,bin_ind] - 0.5)
        return x


class roundModel(nn.Module):
    """
    Learnable model to round integer variables
    """
    def __init__(self, layers, param_keys, var_keys, output_keys=[],
                 int_ind=defaultdict(list), bin_ind=defaultdict(list),
                 continuous_update=False, tolerance=1e-3, name="Rounding"):
        super(roundModel, self).__init__()
        # data keys
        self.param_keys, self.var_keys = param_keys, var_keys
        self.input_keys = self.param_keys + self.var_keys
        self.output_keys = output_keys if output_keys else self.var_keys
        # index of integer and binary variables
        self.int_ind = int_ind
        self.bin_ind = bin_ind
        # update continuous or not
        self.continuous_update = continuous_update
        # numerical tolerance
        self.tolerance = tolerance
        # autograd functions
        self.floor, self.bin = diffFloor(), diffBinarize()
        # sequence
        self.layers = layers
        # name
        self.name = name

    def forward(self, data):
        # sigmoid binary variables
        # self._sigmoid(data)
        # get vars & params
        p, x = self._extract_data(data)
        # concatenate all features: params + sol
        f = torch.cat(p+x, dim=-1)
        # forward
        h = self.layers(f)
        # rounding
        output_data = self._process_rounding(h, data)
        return output_data

    def _sigmoid(self, data):
        for k in self.var_keys:
            data[k][:,self.bin_ind[k]] = torch.sigmoid(data[k][:,self.bin_ind[k]])

    def _extract_data(self, data):
        p = [data[k] for k in self.param_keys]
        x = [data[k] for k in self.var_keys]
        return p, x

    def _process_rounding(self, h, data):
        output_data = {}
        for k_in, k_out in zip(self.var_keys, self.output_keys):
            # get rounding
            x_rnd = self._round_vars(h, data, k_in)
            output_data[k_out] = x_rnd
            # cut off used h
            h = h[:,data[k_in].shape[1]+1:]
        return output_data

    def _round_vars(self, h, data, key):
        # get index
        int_ind = self.int_ind[key]
        bin_ind = self.bin_ind[key]
        # load x
        x = data[key].clone()
        ###################### integer ######################
        # floor(x)
        x_flr = self.floor(x[:,int_ind])
        # bin(h): binary 0 for floor, 1 for ceil
        bnr = self.bin(h[:,int_ind])
        # mask if already integer
        bnr = self._int_mask(bnr, x[:, int_ind])
        # update continuous variables or not
        if self.continuous_update:
            x_rnd = x + h
        else:
            x_rnd = x
        # update rounding for integer variables int(x) = floor(x) + bin(h)
        x_rnd[:, int_ind] = x_flr + bnr
        ###################### binary ######################
        # update rounding for binary variables: bin(x) = bin(h)
        x_rnd[:, bin_ind] = self.bin(h[:, bin_ind])
        return x_rnd

    def _int_mask(self, bnr, x):
        # difference
        diff_fl = x - torch.floor(x)
        diff_cl = torch.ceil(x) - x
        # mask
        bnr[diff_fl < self.tolerance] = 0.0
        bnr[diff_cl < self.tolerance] = 1.0
        return bnr

    def freeze(self):
        """
        Freezes the parameters of the callable in this node
        """
        for param in self.layers.parameters():
            param.requires_grad = False

    def unfreeze(self):
        """
        Unfreezes the parameters of the callable in this node
        """
        for param in self.layers.parameters():
            param.requires_grad = True


class roundGumbelModel(roundModel):
    """
    Learnable model to round integer variables with Gumbel-Softmax trick
    """
    def __init__(self, layers, param_keys, var_keys, output_keys=[],
                 int_ind=defaultdict(list), bin_ind=defaultdict(list),
                 continuous_update=False, temperature=1.0, tolerance=1e-3, name="Rounding"):
        super(roundGumbelModel, self).__init__(layers, param_keys, var_keys,
                                               output_keys, int_ind, bin_ind,
                                               continuous_update, tolerance, name)
        # random temperature
        self.temperature = temperature
        # binarize
        self.bin = diffGumbelBinarize(temperature=self.temperature)


class roundThresholdModel(roundModel):
    """
    Learnable model to round integer variables with variable threshold
    """
    def __init__(self, layers, param_keys, var_keys,output_keys=[],
                 int_ind=defaultdict(list), bin_ind=defaultdict(list),
                 continuous_update=False, slope=10, name="Rounding"):
        super(roundThresholdModel, self).__init__(layers, param_keys, var_keys,
                                                  output_keys, int_ind, bin_ind,
                                                  continuous_update,
                                                  tolerance=None, name=name)
        # slope
        self.slope= slope
        # binarize
        self.bin = thresholdBinarize(slope=self.slope)

    def _round_vars(self, h, data, key):
        # get index
        int_ind = self.int_ind[key]
        bin_ind = self.bin_ind[key]
        # load x
        x = data[key].clone()
        # get threshold from sigmoid
        threshold = torch.sigmoid(h)
        ###################### integer ######################
        # floor(x)
        x_flr = self.floor(x[:,int_ind])
        # extract fractional part
        x_frc = x[:,int_ind] - x_flr
        # get threshold
        v = threshold[:,int_ind]
        # bin(x, v): binary 0 for floor, 1 for ceil
        bnr = self.bin(x_frc, v)
        # update continuous variables or not
        if self.continuous_update:
            x_rnd = x + h
        else:
            x_rnd = x
        # update rounding for integer variables int(x) = floor(x) + bin(h)
        x_rnd[:, int_ind] = x_flr + bnr
        ###################### binary ######################
        # get threshold
        v = threshold[:,bin_ind]
        # fractional part is itself
        x_frc = x[:,bin_ind]
        # bin(x, v): binary 0 for floor, 1 for ceil
        bnr = self.bin(x_frc, v)
        # update rounding for binary variables int(x) = bin(x, v)
        x_rnd[:, bin_ind] = bnr
        return x_rnd


if __name__ == "__main__":

    import numpy as np

    # random seed
    np.random.seed(42)
    torch.manual_seed(42)

    # init
    num_data = 5000   # number of data
    test_size = 1000  # number of test size
    val_size = 1000   # number of validation size

    # data sample from uniform distribution
    p_low, p_high = 0.0, 1.0
    p_samples = torch.FloatTensor(num_data, 2).uniform_(p_low, p_high)
    data = {"p":p_samples}
    # data split
    from src.utils import data_split
    data_train, data_test, data_dev = data_split(data, test_size=test_size, val_size=val_size)
    # torch dataloaders
    from torch.utils.data import DataLoader
    loader_train = DataLoader(data_train, batch_size=32, num_workers=0,
                              collate_fn=data_train.collate_fn, shuffle=True)
    loader_test  = DataLoader(data_test, batch_size=32, num_workers=0,
                              collate_fn=data_test.collate_fn, shuffle=True)
    loader_dev   = DataLoader(data_dev, batch_size=32, num_workers=0,
                              collate_fn=data_dev.collate_fn, shuffle=True)

    # define neural architecture for the solution map
    import neuromancer as nm
    func = nm.modules.blocks.MLP(insize=2, outsize=4, bias=True,
                                 linear_map=nm.slim.maps["linear"],
                                 nonlin=nn.ReLU, hsizes=[10]*4)
    smap = nm.system.Node(func, ["p"], ["x"], name="smap")

    # define rounding model
    from src.func.layer import netFC
    layers_rnd = netFC(input_dim=6, hidden_dims=[20]*3, output_dim=4)
    #round_func = roundModel(layers=layers_rnd, param_keys=["p"], var_keys=["x"], output_keys=["x_rnd"],
    #                        int_ind={"x":[2,3]}, continuous_update=False, name="round")
    round_func = roundGumbelModel(layers=layers_rnd, param_keys=["p"], var_keys=["x"], output_keys=["x_rnd"],
                                  bin_ind={"x":[2,3]}, continuous_update=False, name="round")
    #round_func = roundThresholdModel(layers=layers_rnd, param_keys=["p"], var_keys=["x"], output_keys=["x_rnd"],
    #                                 int_ind={"x":[2,3]}, continuous_update=False, name="round")


    # build neuromancer problem
    components = nn.ModuleList([smap, round_func])
    from src.problem import nmQuadratic
    loss_fn = nmQuadratic(["p", "x"])

    # training
    lr = 0.001    # step size for gradient descent
    epochs = 200  # number of training epochs
    warmup = 20   # number of epochs to wait before enacting early stopping policy
    patience = 20 # number of epochs with no improvement in eval metric to allow before early stopping
    # set adamW as optimizer
    optimizer = torch.optim.AdamW(components.parameters(), lr=lr)
    # training
    from src.problem.neuromancer.trainer import trainer
    my_trainer = trainer(components, loss_fn, optimizer, patience, warmup)
    my_trainer.train(loader_train, loader_dev, epochs)
    print()

    # init mathematical model
    from src.problem.math_solver.quadratic import quadratic
    model = quadratic()

    # test neuroMANCER
    p = 0.6, 0.8
    datapoint = {"p": torch.tensor([[*p]], dtype=torch.float32),
                 "name":"test"}
    model.set_param_val({"p":p})
    from src.utils import nm_test_solve, ms_test_solve
    print("SCIP:")
    ms_test_solve(model)
    print("neuroMANCER:")
    nm_test_solve("x_rnd", components, datapoint, model)
