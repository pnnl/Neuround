"""
neuround: Neuromancer extension for Mixed-Integer Nonlinear Programming.
"""

# ---- Re-exports from neuromancer ----
from neuromancer.modules.blocks import MLP
from neuromancer.system import Node
from neuromancer.dataset import DictDataset
from neuromancer.trainer import Trainer
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
    "MLP", "Node", "DictDataset", "Trainer",
    "Objective", "Constraint", "PenaltyLoss", "Problem",
    # neuround modules
    "MLPBnDrop",
    "VarType", "variable",
    "GradientProjection",
    "LearnableSolver",
]
