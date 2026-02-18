"""
neuround: Neuromancer extension for Mixed-Integer Nonlinear Programming.
"""

# ---- Re-exports from neuromancer ----
from neuromancer.system import Node
from neuromancer.dataset import DictDataset
from neuromancer.constraint import Objective, Constraint
from neuromancer.loss import PenaltyLoss
from neuromancer.problem import Problem

# ---- neuround modules ----
from neuround.blocks import MLPBnDrop
from neuround.variable import VarType, variable
from neuround.projection import GradientProjection
from neuround.solver import LearnableSolver

__all__ = [
    # neuromancer re-exports
    "Node", "DictDataset", "Trainer",
    "Objective", "Constraint", "PenaltyLoss", "Problem",
    # neuround modules
    "MLPBnDrop",
    "VarType", "variable",
    "GradientProjection",
    "LearnableSolver",
]


def __getattr__(name):
    if name == "Trainer":
        from neuromancer.trainer import Trainer as _BaseTrainer
        from neuround.solver import fast

        class Trainer(_BaseTrainer):
            """Trainer with implicit torch.compile + AMP autocast."""
            def __init__(self, problem, *args, device="cpu", compile=True, **kwargs):
                super().__init__(fast(problem, device, compile=compile), *args,
                                 device=device, **kwargs)

        return Trainer
    raise AttributeError(f"module 'neuround' has no attribute {name!r}")
