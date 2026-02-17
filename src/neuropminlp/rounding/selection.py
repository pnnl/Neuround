"""
Adaptive selection rounding layers (deterministic and stochastic).
"""

import torch

from neuropminlp.rounding.base import RoundingNode
from neuropminlp.rounding.functions import (
    DiffFloor, DiffBinarize, DiffGumbelBinarize,
)


class AdaptiveSelectionRounding(RoundingNode):
    """
    Adaptive selection rounding with network-adjusted variables.

    Network selects rounding direction for integer/binary variables.
    Uses deterministic STE binarization.

    Args:
        vars: Variable or list of variables with type metadata.
        param_keys: List of parameter keys to read from data dict.
        net: Network mapping [params, vars] to per-variable selection.
        continuous_update: Whether to update continuous variables (default: False).
        name: Module name.
    """

    def __init__(self, vars, param_keys, net,
                 continuous_update=False, tolerance=1e-3,
                 name="adaptive_selection_rounding"):
        super().__init__(vars, name)
        self.param_keys = param_keys
        self.continuous_update = continuous_update
        self.tolerance = tolerance

        # Extend input keys to include parameter keys
        self.input_keys = list(param_keys) + self.input_keys

        # Network: [params, vars] -> per-variable selection
        self.net = net

        # Differentiable floor via STE
        self.floor = DiffFloor()
        # Deterministic STE binarization
        self.binarize = DiffBinarize()

    def forward(self, data):
        # Concatenate all problem parameters
        params = torch.cat([data[k] for k in self.param_keys], dim=-1)
        # Concatenate all relaxed variables
        vars = torch.cat([data[v.relaxed.key] for v in self.vars], dim=-1)
        # Network input: [params, vars]
        features = torch.cat([params, vars], dim=-1)
        hidden = self.net(features)

        # Round per variable using offset tracking
        output = {}
        offset = 0
        for var in self.vars:
            n = var.num_vars
            x = data[var.relaxed.key].clone()
            h_var = hidden[:, offset:offset + n]

            # Optionally update continuous variables via network adjustment
            if self.continuous_update and var.continuous_indices:
                x[:, var.continuous_indices] += h_var[:, var.continuous_indices]

            # Round integer variables: floor(x) + binarize(h)
            if var.integer_indices:
                x_int = x[:, var.integer_indices]
                # Differentiable floor
                x_floor = self.floor(x_int)
                # Network selects round up or down
                binary = self.binarize(h_var[:, var.integer_indices])
                # Mask if already integer
                binary = self._int_mask(binary, x_int)
                x[:, var.integer_indices] = x_floor + binary

            # Round binary variables: binarize(h)
            if var.binary_indices:
                x[:, var.binary_indices] = self.binarize(
                    h_var[:, var.binary_indices]
                )

            # Store rounded result
            output[var.key] = x
            offset += n
        return output

    def _int_mask(self, binary, x):
        """Mask rounding for values already close to an integer."""
        frac_floor = x - torch.floor(x)
        frac_ceil = torch.ceil(x) - x
        binary[frac_floor < self.tolerance] = 0.0
        binary[frac_ceil < self.tolerance] = 1.0
        return binary


class StochasticAdaptiveSelectionRounding(AdaptiveSelectionRounding):
    """
    Stochastic adaptive selection rounding with Gumbel-Softmax noise.

    Same as AdaptiveSelectionRounding but uses DiffGumbelBinarize
    for stochastic training and deterministic evaluation.

    Args:
        vars: Variable or list of variables with type metadata.
        param_keys: List of parameter keys to read from data dict.
        net: Network mapping [params, vars] to per-variable selection.
        continuous_update: Whether to update continuous variables (default: False).
        temperature: Gumbel-Softmax temperature (default: 1.0).
        name: Module name.
    """

    def __init__(self, vars, param_keys, net,
                 continuous_update=False, temperature=1.0,
                 name="stochastic_adaptive_selection_rounding"):
        super().__init__(vars, param_keys, net,
                         continuous_update=continuous_update,
                         name=name)
        # Replace deterministic STE binarization with Gumbel-Softmax version
        self.binarize = DiffGumbelBinarize(temperature=temperature)
