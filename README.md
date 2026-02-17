# NeuRound

![Framework](img/pipeline.png)

A Learning-to-Optimize (L2O) framework for Mixed-Integer Nonlinear Programming (MINLP), built on top of [NeuroMANCER](https://pnnl.github.io/neuromancer).

Based on the paper: **"[Learning to Optimize for Mixed-Integer Nonlinear Programming with Feasibility Guarantees](https://arxiv.org/abs/2410.11061)"**

## Overview

Neuround solves **parametric MINLP**: given a family of optimization problems that share the same structure but differ in parameter values (e.g., constraint right-hand sides), it learns a neural network that maps parameters directly to high-quality integer solutions, without invoking a traditional solver at inference time.

The key components are differentiable **integer correction layers** (for rounding continuous relaxations to integers) and **gradient-based feasibility projection** (for enforcing feasibility post-hoc). Training is self-supervised: only sampled parameter values are needed, not expensive optimal solutions. The framework scales to problems with tens of thousands of variables at subsecond inference with fast training.

## Citation

```bibtex
@article{tang2024learning,
  title={Learning to optimize for mixed-integer non-linear programming with feasibility guarantees},
  author={Tang, Bo and Khalil, Elias B and Drgo{\v{n}}a, J{\'a}n},
  journal={arXiv preprint arXiv:2410.11061},
  year={2024}
}
```

## Slides

Our talk at ICS 2025. View the slides [here](https://github.com/pnnl/L2O-pMINLP/blob/master/slides/L2O-MINLP.pdf).


## Installation

```bash
pip install -e .
```

**Requirements:** Python >= 3.10, PyTorch, NeuroMANCER


## Tutorial

This tutorial walks through building a learnable solver for a parametric integer quadratic program:

$$\min_{x} \quad \frac{1}{2} x^\top Q x + p^\top x \quad \text{s.t.} \quad Ax \leq b, \quad x \in \mathbb{Z}^n$$

Here $Q$, $p$, $A$ are fixed problem coefficients, while $b$ is the **varying parameter**. Different values of $b$ define different problem instances. The goal of Neuround is to learn a neural network mapping $b \mapsto x^*$ so that, given any new $b$, the network efficiently predicts a high-quality integer solution.

The training pipeline:
1. **Sample** a dataset of parameter values $\{b^{(i)}\}$ (no optimal solutions needed)
2. **Train** the network end-to-end with a self-supervised penalty loss (objective + constraint violations)
3. **Predict** solutions for unseen parameters via a single forward pass


### Step 1: Define Variables

Use `variable()` to create decision variables with type metadata. This tells rounding layers which indices need integrality enforcement.

```python
from neuround import variable, VarType

# Pure integer variable
x = variable("x", num_vars=5, integer_indices=[0, 1, 2, 3, 4])

# Mixed-integer: indices 0,1 are integer, index 2 is binary, rest continuous
y = variable("y", num_vars=5, integer_indices=[0, 1], binary_indices=[2])
# Equivalent using explicit type list
y = variable("y", var_types=[
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
x.relaxed             # continuous variable with key "x_rel"
x.relaxed.key         # "x_rel"
```

### Step 2: Build Solution Mapping Network

The solution mapping network learns the mapping $b \mapsto x_{\text{rel}}$: it takes problem parameters as input and outputs a continuous relaxation of the solution. Wrap any PyTorch module in a `Node` to integrate it into the pipeline.

```python
from torch import nn
from neuround import MLP, MLPBnDrop, Node

num_var = 5
num_ineq = 5

# Option A: Use MLP
smap_net = MLP(
    insize=num_ineq,         # input: problem parameters (b)
    outsize=num_var,         # output: relaxed solution (x_rel)
    bias=True,
    nonlin=nn.ReLU,
    hsizes=[64] * 4,         # 4 hidden layers of width 64
)

# Option B: Use MLPBnDrop (with BatchNorm + Dropout)
smap_net = MLPBnDrop(
    insize=num_ineq,
    outsize=num_var,
    hsizes=[64] * 4,
    dropout=0.2,
    bnorm=True,
)

# Wrap as a Node: input key "b", output key must match x.relaxed.key
smap = Node(smap_net, ["b"], [x.relaxed.key], name="smap")
```

`Node` specifies the data flow: `data["b"] -> smap_net -> data["x_rel"]`.


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

Neuround uses operator overloading to define objectives and constraints symbolically. The `variable()` calls here create symbolic placeholders — their keys (`"x"`, `"b"`) must match the keys produced by the solution map and the data dict, respectively. They are combined into a `PenaltyLoss` that serves as the self-supervised training signal.

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

`LearnableSolver` composes the solution map, rounding layer, and loss. It automatically validates key/dimension alignment. By default, it builds a `GradientProjection` (1000 steps) for feasibility enforcement at inference.

```python
from neuround import LearnableSolver

# Default: projection enabled (projection_steps=1000)
solver = LearnableSolver(
    smap_node=smap,
    rounding_node=rounding,
    loss=loss,
)

# Disable projection
solver = LearnableSolver(
    smap_node=smap,
    rounding_node=rounding,
    loss=loss,
    projection_steps=0,          # 0 = no projection at inference
)

# Custom projection settings
solver = LearnableSolver(
    smap_node=smap,
    rounding_node=rounding,
    loss=loss,
    projection_steps=500,
    projection_step_size=0.05,
    projection_decay=0.99,       # step size decays each iteration
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

Given new parameter values $b$, the trained solver predicts integer solutions via a forward pass (plus projection if configured). The input can be a batch of instances.

```python
b_test = data_test.datadict["b"].to("cuda")
result = solver.predict({"b": b_test, "name": "test"})

print(result["x"])       # integer solution
print(result["x_rel"])   # continuous relaxation
```

## Package Structure

```
src/neuround/                    # Core package
├── __init__.py                  # Public API
├── variable.py                  # VarType enum & variable() factory
├── blocks.py                    # MLPBnDrop (MLP with BatchNorm + Dropout)
├── solver.py                    # LearnableSolver wrapper
├── rounding/                    # Integer rounding layers
│   ├── functions.py             # Differentiable STE primitives
│   ├── base.py                  # RoundingNode abstract base class
│   ├── ste.py                   # STERounding, StochasticSTERounding
│   ├── threshold.py             # DynamicThresholdRounding, StochasticDynamicThresholdRounding
│   └── selection.py             # AdaptiveSelectionRounding, StochasticAdaptiveSelectionRounding
├── projection/                  # Feasibility projection
│   └── gradient.py              # GradientProjection
└── utils/
experiments/                     # Benchmark experiments (not part of the package)
├── quadratic.py                 # Integer Quadratic Programming (IQP)
├── nonconvex.py                 # Integer Non-Convex Programming (INP)
├── rosenbrock.py                # Mixed-Integer Rosenbrock (MIRB)
├── heuristic.py                 # Rounding heuristics (naive_round, floor_round)
└── math_solver/                 # Pyomo-based exact solvers for evaluation
    ├── abc_solver.py            # Abstract parametric solver base class
    ├── quadratic.py             # IQP solver (Gurobi)
    ├── nonconvex.py             # INP solver (SCIP)
    └── rosenbrock.py            # MIRB solver (SCIP)
tests/                           # pytest test suite
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


## License

MIT
