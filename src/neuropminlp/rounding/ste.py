"""
Straight-Through Estimator based rounding layers (deterministic and stochastic).
"""

from neuropminlp.rounding.base import RoundingNode
from neuropminlp.rounding.functions import DiffFloor, DiffBinarize, DiffGumbelBinarize


class STERounding(RoundingNode):
    """
    STE-based rounding without learnable parameters.

    Integer variables: floor(x) + binarize(fractional - 0.5).
    Binary variables: binarize(x - 0.5).
    Uses deterministic STE binarization.

    Args:
        vars: Variable or list of variables with type metadata.
        name: Module name.
    """

    def __init__(self, vars, name="ste_rounding"):
        super().__init__(vars, name)
        # Differentiable floor via STE
        self.floor = DiffFloor()
        # Deterministic STE binarization
        self.binarize = DiffBinarize()

    def forward(self, data):
        output = {}
        # Round each variable independently (no cross-variable dependency)
        for var in self.vars:
            x = data[var.relaxed.key].clone()

            # Round integer variables: floor(x) + binarize(frac - 0.5)
            if var.integer_indices:
                x_int = x[:, var.integer_indices]
                # Differentiable floor
                x_floor = self.floor(x_int)
                # Binarize fractional part (detach floor to avoid double gradient)
                binary = self.binarize(x_int - x_floor.detach() - 0.5)
                x[:, var.integer_indices] = x_floor + binary

            # Round binary variables: binarize(x - 0.5)
            if var.binary_indices:
                x[:, var.binary_indices] = self.binarize(
                    x[:, var.binary_indices] - 0.5
                )

            # Store rounded result
            output[var.key] = x
        return output


class StochasticSTERounding(STERounding):
    """
    Stochastic STE-based rounding without learnable parameters.

    Same as STERounding but uses DiffGumbelBinarize
    for stochastic training and deterministic evaluation.

    Args:
        vars: Variable or list of variables with type metadata.
        temperature: Gumbel-Softmax temperature (default: 1.0).
        name: Module name.
    """

    def __init__(self, vars, temperature=1.0,
                 name="stochastic_ste_rounding"):
        super().__init__(vars, name=name)
        # Replace deterministic STE binarization with Gumbel-Softmax version
        self.binarize = DiffGumbelBinarize(temperature=temperature)
