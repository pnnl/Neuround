"""
NeuroPMINLP: Neuromancer extension for Mixed-Integer Nonlinear Programming.
"""

# ---- Re-exports from neuromancer ----
from neuromancer.modules.blocks import MLP
from neuromancer.system import Node
from neuromancer.dataset import DictDataset
from neuromancer.trainer import Trainer
from neuromancer.constraint import Objective, Constraint
from neuromancer.loss import PenaltyLoss
from neuromancer.problem import Problem

# ---- neuropminlp modules ----
from neuropminlp.blocks import MLPBnDrop
from neuropminlp.variable import VarType, variable
from neuropminlp.projection import GradientProjection

__all__ = [
    # neuromancer re-exports
    "MLP", "Node", "DictDataset", "Trainer",
    "Objective", "Constraint", "PenaltyLoss", "Problem",
    # neuropminlp modules
    "MLPBnDrop",
    "VarType", "variable",
    "GradientProjection",
]
