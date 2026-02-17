# Neuround

A Learning-to-Optimize (L2O) framework for Mixed-Integer Nonlinear Programming (MINLP), built on top of [NeuroMANCER](https://pnnl.github.io/neuromancer).

Based on the paper: **"[Learning to Optimize for Mixed-Integer Nonlinear Programming](https://arxiv.org/abs/2410.11061)"**

![Framework](img/pipeline.png)


## Overview

Neuround solves **parametric MINLP**: given a family of optimization problems that share the same structure but differ in parameter values (e.g., constraint right-hand sides), it learns a neural network that maps parameters directly to high-quality integer solutions — without invoking a traditional solver at inference time.

The key components are differentiable **integer correction layers** (for rounding continuous relaxations to integers) and **gradient-based feasibility projection** (for enforcing constraints post-hoc). Training is self-supervised: only sampled parameter values are needed, not optimal solutions. The framework scales to problems with tens of thousands of variables at subsecond inference.


## Installation

```bash
pip install -e .
```

**Requirements:** Python >= 3.10, PyTorch, NeuroMANCER


## Tutorial

This tutorial walks through building a learnable solver for a parametric integer quadratic program:

$$\min_{x} \quad \frac{1}{2} x^\top Q x + p^\top x \quad \text{s.t.} \quad Ax \leq b, \quad x \in \mathbb{Z}^n$$

Here $Q$, $p$, $A$ are fixed problem coefficients, while $b$ is the **varying parameter**. Different values of $b$ define different problem instances. The goal of Neuround is to learn a neural network mapping $b \mapsto x^*$ so that, given any new $b$, the network predicts a high-quality integer solution in subsecond time — without invoking a traditional solver.

The training pipeline:
1. **Sample** a dataset of parameter values $\{b^{(i)}\}$ (no optimal solutions needed)
2. **Train** the network end-to-end with a self-supervised penalty loss (objective + constraint violations)
3. **Predict** solutions for unseen parameters via a single forward pass


### Step 1: Define Variables

Use `variable()` to create decision variables with type metadata. This tells rounding layers which indices need integrality enforcement.

```python
from neuround import variable, VarType

# All-integer variable (5 dimensions)
x = variable("x", num_vars=5, integer_indices=[0, 1, 2, 3, 4])

# Mixed-integer: indices 0,1 are integer, index 2 is binary, rest continuous
y = variable("y", num_vars=5, integer_indices=[0, 1], binary_indices=[2])

# Equivalent using explicit type list
z = variable("z", var_types=[
    VarType.INTEGER, VarType.INTEGER, VarType.BINARY,
    VarType.CONTINUOUS, VarType.CONTINUOUS,
])
```

Typed variables automatically gain useful attributes:
```python
x.num_vars            # 5
x.integer_indices     # [0, 1, 2, 3, 4]
x.binary_indices      # []
x.continuous_indices  # []
x.relaxed             # neuromancer variable with key "x_rel"
x.relaxed.key         # "x_rel"
```

The `.relaxed` attribute is key: the solution mapping network outputs `"x_rel"` (continuous relaxation), and rounding layers convert it to `"x"` (integer solution).


### Step 2: Build Solution Mapping Network

The solution mapping network learns the mapping $b \mapsto x_{\text{rel}}$: it takes problem parameters as input and outputs a continuous relaxation of the solution. Use any PyTorch module wrapped in a NeuroMANCER `Node`.

```python
from torch import nn
import neuromancer as nm
from neuround import MLP, MLPBnDrop, Node

num_var = 5
num_ineq = 5

# Option A: Use neuromancer MLP
smap_net = MLP(
    insize=num_ineq,         # input: problem parameters (b)
    outsize=num_var,         # output: relaxed solution (x_rel)
    bias=True,
    linear_map=nm.slim.maps["linear"],
    nonlin=nn.ReLU,
    hsizes=[64] * 4,         # 4 hidden layers of width 64
)

# Option B: Use MLPBnDrop for regularization
smap_net = MLPBnDrop(
    insize=num_ineq,
    outsize=num_var,
    hsizes=[64] * 4,
    dropout=0.2,             # dropout after each hidden layer
    bnorm=True,              # batch normalization
)

# Wrap as a Node: input key "b", output key must match x.relaxed.key
smap = Node(smap_net, ["b"], [x.relaxed.key], name="smap")
```

The `Node` specifies the data flow: `data["b"] -> smap_net -> data["x_rel"]`.


### Step 3: Choose a Rounding Layer

Rounding layers convert continuous relaxations to integer solutions. All inherit from `RoundingNode` and read from `"x_rel"` to produce `"x"`.

**Non-learnable (baseline):**
```python
from neuround.rounding import STERounding

rounding = STERounding(x)
```

**Learnable (recommended):** these use a secondary network that takes the concatenation of problem parameters $b$ and the relaxed solution $x_{\text{rel}}$, and predicts per-variable rounding decisions.

```python
from neuround.rounding import (
    DynamicThresholdRounding,
    StochasticAdaptiveSelectionRounding,
)

# Rounding network: [b, x_rel] -> per-variable rounding decisions
rnd_net = MLPBnDrop(
    insize=num_ineq + num_var,   # params (b) + relaxed vars (x_rel)
    outsize=num_var,
    hsizes=[64] * 3,
    dropout=0,
    bnorm=False,
)

# Learnable Thresholding (LT): learns per-variable rounding thresholds
rounding = DynamicThresholdRounding(
    vars=x,
    param_keys=["b"],            # which data keys are problem parameters
    net=rnd_net,
    continuous_update=False,     # whether to also adjust continuous vars via net
)

# Rounding Classification (RC): learns rounding direction per variable
rounding = StochasticAdaptiveSelectionRounding(
    vars=x,
    param_keys=["b"],
    net=rnd_net,
    continuous_update=True,
)
```


### Step 4: Define Loss (Objectives + Constraints)

Neuround uses NeuroMANCER's operator overloading to define objectives and constraints symbolically. The `variable()` calls here create symbolic placeholders — their keys (`"x"`, `"b"`) must match the keys produced by the solution map and the data dict respectively. They are combined into a `PenaltyLoss` that serves as the self-supervised training signal.

```python
import torch
import numpy as np
from neuround import variable, PenaltyLoss

# Fixed problem coefficients (Q, p, A do not change across instances)
rng = np.random.RandomState(17)
Q = torch.from_numpy(0.01 * np.diag(rng.random(size=num_var))).float()
p = torch.from_numpy(0.1 * rng.random(num_var)).float()
A = torch.from_numpy(rng.normal(scale=0.1, size=(num_ineq, num_var))).float()

# Symbolic variables for loss expression
# "x" matches the rounding layer output; "b" matches the data dict key
x_sym = variable("x")
b_sym = variable("b")

# Objective: minimize (1/2) x^T Q x + p^T x
f = 0.5 * torch.sum((x_sym @ Q) * x_sym, dim=1) + torch.sum(p * x_sym, dim=1)
obj = f.minimize(weight=1.0, name="obj")

# Constraint: Ax <= b (with penalty weight)
penalty_weight = 100
con = penalty_weight * (x_sym @ A.T <= b_sym)

# Combine into loss
loss = PenaltyLoss(objectives=[obj], constraints=[con])
```


### Step 5: Assemble the Solver

`LearnableSolver` composes the solution map, rounding layer, and loss. It automatically validates key/dimension alignment and optionally builds a `GradientProjection` for inference.

```python
from neuround import LearnableSolver

# Without projection
solver = LearnableSolver(
    smap_node=smap,
    rounding_node=rounding,
    loss=loss,
)

# With projection (for feasibility enforcement at inference)
solver = LearnableSolver(
    smap_node=smap,
    rounding_node=rounding,
    loss=loss,
    projection_steps=1000,       # max gradient projection iterations
    projection_step_size=0.01,   # step size for gradient descent
    projection_decay=1.0,        # step size decay factor (1.0 = no decay)
)
```


### Step 6: Prepare Data & Train

Since this is parametric optimization, training data consists of sampled parameter values — each $b^{(i)}$ defines one problem instance. No optimal solutions are needed; the penalty loss in Step 4 provides the training signal.

```python
from torch.utils.data import DataLoader
from neuround import DictDataset

# Sample parameter values (each b defines a different problem instance)
num_data = 10000
b_samples = torch.from_numpy(
    np.random.uniform(-1, 1, size=(num_data, num_ineq))
).float()

# Split into train/val/test
data_train = DictDataset({"b": b_samples[:8000]}, name="train")
data_val   = DictDataset({"b": b_samples[8000:9000]}, name="val")
data_test  = DictDataset({"b": b_samples[9000:]}, name="test")

loader_train = DataLoader(data_train, batch_size=64, shuffle=True,
                          collate_fn=data_train.collate_fn)
loader_val   = DataLoader(data_val, batch_size=64, shuffle=False,
                          collate_fn=data_val.collate_fn)

# Train
optimizer = torch.optim.AdamW(solver.problem.parameters(), lr=1e-3)
solver.train(
    loader_train, loader_val, optimizer,
    epochs=200,      # max epochs
    patience=20,     # early stopping patience
    warmup=20,       # warmup epochs before early stopping
    device="cuda",
)
```


### Step 7: Predict

Given a new parameter value $b$, the trained solver predicts an integer solution via a single forward pass (plus optional projection).

```python
# Single-instance prediction (with projection if configured)
b_test = torch.unsqueeze(data_test.datadict["b"][0], 0).to("cuda")
result = solver.predict({"b": b_test, "name": "test"})

print(result["x"])       # integer solution
print(result["x_rel"])   # continuous relaxation

# Without projection (manual forward pass)
solver.problem.eval()
with torch.no_grad():
    data = {"b": b_test, "name": "test"}
    data.update(solver.smap_node(data))       # b -> x_rel
    data.update(solver.rounding_node(data))   # x_rel -> x
print(data["x"])
```


## Examples

### Example 1: Comparing Rounding Methods

The tutorial above uses `DynamicThresholdRounding`. Here we show how to swap in different rounding strategies — everything else stays the same.

```python
from neuround import MLPBnDrop, LearnableSolver
from neuround.rounding import (
    STERounding,
    DynamicThresholdRounding,
    StochasticAdaptiveSelectionRounding,
)

# Assume x, smap, loss, num_ineq, num_var are defined as in the tutorial

# ── Option A: STE rounding (no learnable params) ──
rounding_ste = STERounding(x)

# ── Option B: Learnable thresholding (LT) ──
rnd_net_lt = MLPBnDrop(insize=num_ineq + num_var, outsize=num_var,
                       hsizes=[64] * 4, dropout=0, bnorm=False)
rounding_lt = DynamicThresholdRounding(
    vars=x, param_keys=["b"], net=rnd_net_lt)

# ── Option C: Learnable rounding classification (RC) with Gumbel noise ──
rnd_net_rc = MLPBnDrop(insize=num_ineq + num_var, outsize=num_var,
                       hsizes=[64] * 4, dropout=0, bnorm=False)
rounding_rc = StochasticAdaptiveSelectionRounding(
    vars=x, param_keys=["b"], net=rnd_net_rc, continuous_update=True)

# Pick one and build the solver
solver = LearnableSolver(smap, rounding_rc, loss)
```

### Example 2: Mixed-Integer Variables

When a problem has both integer and continuous variables:

```python
from neuround import variable, VarType, MLP, MLPBnDrop, Node, PenaltyLoss, LearnableSolver
from neuround.rounding import StochasticAdaptiveSelectionRounding

# 3 integer + 2 continuous variables
x = variable("x", num_vars=5, integer_indices=[0, 1, 2])
# x.integer_indices    -> [0, 1, 2]
# x.continuous_indices -> [3, 4]

# Solution map
smap = Node(MLP(insize=4, outsize=5, hsizes=[64, 64]),
            ["p"], [x.relaxed.key], name="smap")

# Rounding with continuous_update=True allows the network
# to also adjust continuous variables alongside rounding
rnd_net = MLPBnDrop(insize=4 + 5, outsize=5,
                    hsizes=[64, 64], dropout=0, bnorm=False)
rounding = StochasticAdaptiveSelectionRounding(
    vars=x, param_keys=["p"], net=rnd_net,
    continuous_update=True,   # network also adjusts x[:, 3:5]
)

# ... define loss and solver as before
```

### Example 3: Multiple Variable Groups

When a problem has separate variable groups:

```python
from neuround import variable, MLP, MLPBnDrop, Node
from neuround.rounding import DynamicThresholdRounding

# Two groups of variables
x = variable("x", num_vars=3, integer_indices=[0, 1, 2])
y = variable("y", num_vars=2, binary_indices=[0, 1])

# Solution map outputs both relaxed variables
smap = Node(MLP(insize=4, outsize=5, hsizes=[64, 64]),
            ["p"], [x.relaxed.key, y.relaxed.key], name="smap")
# data["p"] -> smap -> data["x_rel"] (first 3 dims), data["y_rel"] (last 2 dims)

# Rounding handles both variable groups
rnd_net = MLPBnDrop(insize=4 + 5, outsize=5,
                    hsizes=[64, 64], dropout=0, bnorm=False)
rounding = DynamicThresholdRounding(
    vars=[x, y],             # list of variables
    param_keys=["p"],
    net=rnd_net,
)
# Produces data["x"] (rounded integers) and data["y"] (rounded binaries)
```

### Example 4: Low-Level Usage (Without LearnableSolver)

For full control, compose components manually:

```python
import torch
from neuround import variable, MLP, Node, Problem, Trainer, PenaltyLoss, GradientProjection
from neuround.rounding import STERounding

x = variable("x", num_vars=5, integer_indices=list(range(5)))

smap = Node(MLP(insize=5, outsize=5, hsizes=[64, 64]),
            ["b"], [x.relaxed.key], name="smap")
rounding = STERounding(x)

# Manually build Problem
loss = ...  # your PenaltyLoss
problem = Problem(nodes=[smap, rounding], loss=loss)

# Train with neuromancer Trainer directly
problem.to("cuda")
optimizer = torch.optim.AdamW(problem.parameters(), lr=1e-3)
trainer = Trainer(problem, loader_train, loader_val,
                  optimizer=optimizer, epochs=200,
                  patience=20, warmup=20, device="cuda")
best_model = trainer.train()
problem.load_state_dict(best_model)

# Manual inference with projection
projection = GradientProjection(
    rounding_components=[rounding],
    constraints=list(loss.constraints),
    target_keys=[x.relaxed.key],
    num_steps=500,
    step_size=0.01,
)

problem.eval()
data = {"b": b_test, "name": "test"}
with torch.no_grad():
    data.update(smap(data))
data = projection(data)  # gradient-based feasibility projection
print(data["x"])
```


## API Reference

### `variable(key, num_vars=None, integer_indices=None, binary_indices=None, var_types=None)`

Create a NeuroMANCER variable with optional type metadata.

| Parameter | Type | Description |
|---|---|---|
| `key` | `str` | Variable name (used as dictionary key) |
| `num_vars` | `int` | Total number of variables |
| `integer_indices` | `list[int]` | Indices of integer variables |
| `binary_indices` | `list[int]` | Indices of binary variables |
| `var_types` | `list[VarType]` | Explicit type list (mutually exclusive with index params) |

**Returns:** NeuroMANCER variable. If type params are given, additional attributes are attached: `var_types`, `num_vars`, `integer_indices`, `binary_indices`, `continuous_indices`, `relaxed`.


### `VarType`

Enum with values: `VarType.CONTINUOUS`, `VarType.INTEGER`, `VarType.BINARY`.


### `MLPBnDrop(insize, outsize, hsizes, nonlin=ReLU, dropout=0.2, bnorm=True, bias=True)`

MLP with optional BatchNorm and Dropout. Architecture: `[Linear -> nonlin -> BatchNorm -> Dropout] * N -> Linear`.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `insize` | `int` | | Input dimension |
| `outsize` | `int` | | Output dimension |
| `hsizes` | `list[int]` | | Hidden layer sizes |
| `nonlin` | `nn.Module` | `nn.ReLU` | Activation function class |
| `dropout` | `float` | `0.2` | Dropout probability (0 to disable) |
| `bnorm` | `bool` | `True` | Enable BatchNorm |
| `bias` | `bool` | `True` | Use bias in linear layers |


### Rounding Layers

All rounding layers inherit from `RoundingNode`. They read from `data[var.relaxed.key]` (e.g., `"x_rel"`) and output `data[var.key]` (e.g., `"x"`).

#### `STERounding(vars, name="ste_rounding")`

Non-learnable STE-based rounding. Integer: `floor(x) + binarize(frac - 0.5)`. Binary: `binarize(x - 0.5)`.

#### `StochasticSTERounding(vars, temperature=1.0, name="stochastic_ste_rounding")`

Same as `STERounding` but with Gumbel noise for stochastic exploration during training. Deterministic at eval.

#### `DynamicThresholdRounding(vars, param_keys, net, continuous_update=False, slope=10, name=...)`

Learnable threshold-based rounding. The network predicts per-variable thresholds from `[params, relaxed_vars]`. Integer: `floor(x) + threshold_binarize(frac, thresh)`. Binary: `threshold_binarize(x, thresh)`.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `vars` | variable or list | | Typed variable(s) |
| `param_keys` | `list[str]` | | Data keys for problem parameters |
| `net` | `nn.Module` | | Network: `[params, vars]` -> per-variable output |
| `continuous_update` | `bool` | `False` | Also adjust continuous variables via network |
| `slope` | `float` | `10` | Sigmoid slope for threshold binarization |

#### `StochasticDynamicThresholdRounding(vars, param_keys, net, continuous_update=False, temperature=1.0, name=...)`

Same as `DynamicThresholdRounding` but with Gumbel noise. Extra parameter: `temperature` (default `1.0`).

#### `AdaptiveSelectionRounding(vars, param_keys, net, continuous_update=False, tolerance=1e-3, name=...)`

Learnable direction-selection rounding. The network classifies each variable's rounding direction. Integer: `floor(x) + binarize(net_output)`. Binary: `binarize(net_output)`. Includes masking for values already close to integer (within `tolerance`).

| Parameter | Type | Default | Description |
|---|---|---|---|
| `vars` | variable or list | | Typed variable(s) |
| `param_keys` | `list[str]` | | Data keys for problem parameters |
| `net` | `nn.Module` | | Network: `[params, vars]` -> per-variable selection |
| `continuous_update` | `bool` | `False` | Also adjust continuous variables via network |
| `tolerance` | `float` | `1e-3` | Masking threshold for already-integer values |

#### `StochasticAdaptiveSelectionRounding(vars, param_keys, net, continuous_update=False, temperature=1.0, name=...)`

Same as `AdaptiveSelectionRounding` but with Gumbel noise. Extra parameter: `temperature` (default `1.0`).


### `GradientProjection(rounding_components, constraints, target_keys, num_steps=1000, step_size=0.01, decay=1.0, tolerance=1e-6)`

Gradient-based feasibility projection. Iteratively applies gradient descent on constraint violations to the relaxed variables, then re-rounds to preserve integrality.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `rounding_components` | `list` | | Rounding modules to apply at each iteration |
| `constraints` | `list` | | NeuroMANCER `Constraint` objects |
| `target_keys` | `list[str]` | | Relaxed variable keys to adjust (e.g., `["x_rel"]`) |
| `num_steps` | `int` | `1000` | Maximum projection iterations |
| `step_size` | `float` | `0.01` | Gradient descent learning rate |
| `decay` | `float` | `1.0` | Multiplicative step size decay per iteration |
| `tolerance` | `float` | `1e-6` | Convergence threshold (max violation) |


### `LearnableSolver(smap_node, rounding_node, loss, projection_steps=1000, projection_step_size=0.01, projection_decay=1.0)`

End-to-end wrapper composing solution mapping, rounding, and loss into a trainable solver.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `smap_node` | `Node` | | Solution mapping node (params -> relaxed solution) |
| `rounding_node` | `RoundingNode` | | Rounding layer (relaxed -> integer) |
| `loss` | `PenaltyLoss` | | Penalty loss (objectives + constraints) |
| `projection_steps` | `int` | `1000` | Max projection iterations (0 = no projection) |
| `projection_step_size` | `float` | `0.01` | Projection step size |
| `projection_decay` | `float` | `1.0` | Projection step size decay |

**Methods:**

- **`train(loader_train, loader_val, optimizer, epochs=200, patience=20, warmup=20, device="cpu")`** -- Train using NeuroMANCER `Trainer` with early stopping.
- **`predict(data)`** -- Inference: `smap (no_grad) -> [projection (grad)] -> rounding`. Returns updated data dict with both `"x_rel"` and `"x"`.


### NeuroMANCER Re-exports

For convenience:

```python
from neuround import MLP, Node, DictDataset, Trainer
from neuround import Objective, Constraint, PenaltyLoss, Problem
```


## Package Structure

```
src/neuround/
├── __init__.py              # Public API & neuromancer re-exports
├── variable.py              # VarType enum & variable() factory
├── blocks.py                # MLPBnDrop (MLP with BatchNorm + Dropout)
├── solver.py                # LearnableSolver wrapper
├── rounding/                # Integer rounding layers
│   ├── functions.py         # Differentiable STE primitives
│   ├── base.py              # RoundingNode abstract base class
│   ├── ste.py               # STERounding, StochasticSTERounding
│   ├── threshold.py         # DynamicThresholdRounding, StochasticDynamicThresholdRounding
│   └── selection.py         # AdaptiveSelectionRounding, StochasticAdaptiveSelectionRounding
├── projection/              # Feasibility projection
│   └── gradient.py          # GradientProjection
└── utils/
tests/                       # pytest test suite
```


## Architecture

```
Problem Parameters (b)
    |
    v
[Solution Map]       ---> Continuous Relaxation (x_rel)
    |
    v
[Rounding Layer]     ---> Integer Solution (x)
    |
    v                      (optional, inference only)
[Gradient Projection] --> Feasible Integer Solution (x)
    |-- compute constraint violations
    |-- gradient descent on x_rel
    '-- re-round until convergence
```


## Methodology

### Integer Correction Layers

Two learnable correction layers enforce integrality in neural network output:

<div align="center">
    <img src="img/method_RC.png" alt="example for RC" width="48%"/>
    <img src="img/method_LT.png" alt="example for RT" width="48%"/>
</div>

- **Rounding Classification (RC)** / `AdaptiveSelectionRounding`: Learns a classification strategy to determine rounding directions for integer variables.
- **Learnable Thresholding (LT)** / `DynamicThresholdRounding`: Learns a threshold value for each integer variable to decide whether to round up or down.

### Integer Feasibility Projection

A gradient-based projection iteratively refines infeasible solutions. The figure below illustrates how the projection step adjusts a solution over multiple iterations.

<div align="center"> <img src="img/example2.png" alt="Feasibility Projection Iterations" width="40%"/> </div>

By integrating feasibility projection with the correction layers, we extend RC and LT into **RC-P** and **LT-P**, respectively.


## Performance

Our learning-based methods (RC & LT) achieve comparable or superior performance to exact solvers (EX) while being orders of magnitude faster:

<div align="center">
    <img src="img/cq_s100_penalty.png" alt="Penalty Effect on IQP" width="40%"/>
    <img src="img/rb_s100_penalty.png" alt="Penalty Effect on MIRB" width="40%"/>
</div>

With properly tuned penalty weights, the approach attains comparable or better objective values within subsecond, while exact solvers require up to 1000 seconds.


## Reproducibility

Three benchmark problems are included:

- **Integer Quadratic Problems (IQP)**: Convex quadratic objective with linear constraints and integer variables.
- **Integer Non-Convex Problems (INP)**: Non-convex variant with trigonometric terms.
- **Mixed-Integer Rosenbrock Problems (MIRB)**: Rosenbrock function with linear and non-linear constraints.

```bash
python run_qp.py --size 5
python run_nc.py --size 10 --penalty 1 --project
python run_rb.py --size 100 --penalty 10 --project
```

Arguments: `--size` (problem size), `--penalty` (constraint violation weight), `--project` (enable feasibility projection).


## Citation

```bibtex
@article{tang2024learning,
  title={Learning to Optimize for Mixed-Integer Non-linear Programming},
  author={Tang, Bo and Khalil, Elias B and Drgo{\v{n}}a, J{\'a}n},
  journal={arXiv preprint arXiv:2410.11061},
  year={2024}
}
```

## Slides

Our talk at ICS 2025. View the slides [here](https://github.com/pnnl/L2O-pMINLP/blob/master/slides/L2O-MINLP.pdf).


## License

MIT
