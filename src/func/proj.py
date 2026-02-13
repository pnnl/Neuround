"""
Differentiable/Learnable to project
"""
import torch
from torch import nn

from neuromancer.modules.solvers import GradientProjection as gradProj
from neuromancer.gradients import gradient

class solPredGradProj(nn.Module):
    """
    A module to re-predict solutions and then project them onto a feasible region.
    This is done by first re-predicting the solution considering constraints,
    followed by a projection step.
    """
    def __init__(self, constraints, param_keys, var_keys, output_keys=[], layers=None,
                 num_steps=10, step_size=0.1, decay=0.1, residual=True, name="Projection"):
        super(solPredGradProj, self).__init__()
        # data keys
        self.param_keys, self.var_keys = param_keys, var_keys
        self.input_keys = self.param_keys + self.var_keys
        self.output_keys = output_keys if output_keys else [self.var_key]
        # flag to only predict residual of solution
        self.residual = residual
        # list of neuromancer constraints
        self.constraints = constraints
        # projection
        self.num_steps = num_steps
        self.step_size = step_size
        self.decay = decay
        self.gradProj = gradProj(constraints=self.constraints, input_keys=self.output_keys,
                                 output_keys=self.output_keys, num_steps=self.num_steps,
                                 step_size=self.step_size, decay=self.decay)
        # sequence
        self.layers = layers
        # name
        self.name = name

    def forward(self, data):
        # init output data
        for k_in, k_out in zip(self.var_keys, self.output_keys):
            data[k_out] = data[k_in]
        # get grad of violation
        energy, grads = self._cal_violation(data)
        # get vars & params
        p, x = self._extractData(data)
        # concatenate all features: params + sol + grads
        f = torch.cat(p + x + grads, dim=-1)
        # change solution value
        if self.layers is not None:
            h = self.layers(f)
            data = self._update_solutions(data, h)
        # proj
        data = self.gradProj(data)
        return data

    def _extractData(self, data):
        """
        Extract parameters and variables from the input data.
        """
        p = [data[k] for k in self.param_keys]
        x = [data[k] for k in self.var_keys]
        return p, x

    def _cal_violation(self, data):
        """
        Calculate the violation magnitude for constraints.
        """
        violations = []
        # get violation magnitude
        for constr in self.constraints:
            output = constr(data)
            violation = output[constr.output_keys[2]]
            violations.append(violation.reshape(violation.shape[0], -1))
        # calculate average
        energy = torch.mean(torch.abs(torch.cat(violations, dim=-1)), dim=1)
        # get grads
        grads = []
        for k in self.var_keys:
            step = gradient(energy, data[k]).detach()
            grads.append(step)
        return energy, grads

    def _update_solutions(self, data, h):
        """
        Update solutions based on the model output and optionally
        only consider the residual of the solution.
        """
        for k_in, k_out in zip(self.var_keys, self.output_keys):
            # new solution
            if self.residual:
                # update the solution by adding the residual
                data[k_out] = data[k_in] + h[:,:data[k_in].shape[1]]
            else:
                # update the solution by replacing the value
                data[k_out] = data[k_in].clone()
            # cut off used out
            h = h[:,data[k_in].shape[1]:]
        return data

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

    # get objective function & constraints
    from src.problem import nmQuadratic
    obj_bar, constrs_bar = nmQuadratic(["x_bar"], ["p"], penalty_weight=10)
    obj_rnd, constrs_rnd = nmQuadratic(["x_rnd"], ["p"], penalty_weight=10)

    # define neural architecture for the solution map
    import neuromancer as nm
    func = nm.modules.blocks.MLP(insize=2, outsize=4, bias=True,
                                 linear_map=nm.slim.maps["linear"],
                                 nonlin=nn.ReLU, hsizes=[10]*4)
    smap = nm.system.Node(func, ["p"], ["x"], name="smap")

    # define rounding model
    from src.func.layer import netFC
    layers_rnd = netFC(input_dim=6, hidden_dims=[20]*3, output_dim=4)
    from src.func.rnd import roundModel, roundGumbelModel, roundThresholdModel
    #round_func = roundModel(layers=layers_rnd, param_keys=["p"], var_keys=["x"], output_keys=["x_rnd"],
    #                        int_ind={"x":[2,3]}, continuous_update=False, name="round")
    round_func = roundGumbelModel(layers=layers_rnd, param_keys=["p"], var_keys=["x"], output_keys=["x_rnd"],
                                  bin_ind={"x":[2,3]}, continuous_update=False, name="round")
    #round_func = roundThresholdModel(layers=layers_rnd, param_keys=["p"], var_keys=["x"], output_keys=["x_rnd"],,
    #                                 int_ind={"x":[2,3]}, continuous_update=False, name="round")

    # define projection layer
    num_steps = 10
    step_size = 0.1
    decay = 0.1
    #proj = gradProj(constraints=constrs_rnd, input_keys=["x_rnd"], output_keys=["x_bar"],
    #                num_steps=num_steps, step_size=step_size, decay=decay)
    layers_proj = netFC(input_dim=10, hidden_dims=[20]*3, output_dim=4)
    proj = solPredGradProj(layers=layers_proj, constraints=constrs_bar, param_keys=["p"],
                           var_keys=["x_rnd"], output_keys=["x_bar"], num_steps=num_steps,
                           step_size=step_size, decay=decay)

    # build neuromancer problem
    components = [smap, round_func, proj]
    loss = nm.loss.PenaltyLoss(obj_bar, constrs_bar)
    problem = nm.problem.Problem(components, loss, grad_inference=True)

    # training
    lr = 0.001    # step size for gradient descent
    epochs = 200  # number of training epochs
    warmup = 20   # number of epochs to wait before enacting early stopping policy
    patience = 20 # number of epochs with no improvement in eval metric to allow before early stopping
    # set adamW as optimizer
    optimizer = torch.optim.AdamW(problem.parameters(), lr=lr)
    # define trainer
    trainer = nm.trainer.Trainer(
        problem,
        loader_train,
        loader_dev,
        loader_test,
        optimizer,
        epochs=epochs,
        patience=patience,
        warmup=warmup)
    # train solution map
    best_model = trainer.train()
    # load best model dict
    problem.load_state_dict(best_model)
    print()

    # init mathmatic model
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
    print("Init Solution:")
    nm_test_solve(["x"], problem, datapoint, model)
    print("Rounding:")
    nm_test_solve(["x_rnd"], problem, datapoint, model)
    print("Projection:")
    nm_test_solve(["x_bar"], problem, datapoint, model)
