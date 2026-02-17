"""
Dynamic threshold rounding layers (deterministic and stochastic).
"""

import torch

from neuropminlp.rounding.base import RoundingNode
from neuropminlp.rounding.functions import (
    DiffFloor, ThresholdBinarize, GumbelThresholdBinarize,
)


class DynamicThresholdRounding(RoundingNode):
    """
    Dynamic threshold rounding with MLP-predicted thresholds.

    MLP predicts per-variable thresholds from concatenated
    problem parameters and relaxed solutions.

    Args:
        vars: Variable or list of variables with type metadata.
        param_keys: List of parameter keys to read from data dict.
        net: Network mapping [params, vars] to per-variable outputs.
        continuous_update: Whether to update continuous variables (default: False).
        slope: Slope for sigmoid-smoothed binarization (default: 10).
        name: Module name.
    """

    def __init__(self, vars, param_keys, net,
                 continuous_update=False, slope=10,
                 name="dynamic_threshold_rounding"):
        super().__init__(vars, name)
        self.param_keys = param_keys
        self.continuous_update = continuous_update

        # Extend input keys to include parameter keys
        self.input_keys = list(param_keys) + self.input_keys

        # Network: [params, vars] -> per-variable thresholds
        self.net = net

        # Differentiable floor via STE
        self.floor = DiffFloor()
        # Sigmoid-smoothed threshold binarization
        self.threshold_binarize = ThresholdBinarize(slope=slope)

    def forward(self, data):
        # Concatenate all problem parameters
        params = torch.cat([data[k] for k in self.param_keys], dim=-1)
        # Concatenate all relaxed variables
        vars = torch.cat([data[v.relaxed.key] for v in self.vars], dim=-1)
        # Network input: [params, vars]
        features = torch.cat([params, vars], dim=-1)

        # Predict raw outputs and map to [0, 1] thresholds
        hidden = self.net(features)
        thresholds = torch.sigmoid(hidden)

        # Split and round per variable using offset tracking
        output = {}
        offset = 0
        for var in self.vars:
            n = var.num_vars
            x = data[var.relaxed.key].clone()
            # Slice network output for this variable
            h_var = hidden[:, offset:offset + n]
            thresh_var = thresholds[:, offset:offset + n]

            # Optionally update continuous variables via network adjustment
            if self.continuous_update and var.continuous_indices:
                x[:, var.continuous_indices] += h_var[:, var.continuous_indices]

            # Round integer variables: floor(x) + threshold_binarize(frac, thresh)
            if var.integer_indices:
                x_floor = self.floor(x[:, var.integer_indices])
                x_frac = x[:, var.integer_indices] - x_floor
                thresh = thresh_var[:, var.integer_indices]
                binary = self.threshold_binarize(x_frac, thresh)
                x[:, var.integer_indices] = x_floor + binary

            # Round binary variables: threshold_binarize(x, thresh)
            if var.binary_indices:
                thresh = thresh_var[:, var.binary_indices]
                x[:, var.binary_indices] = self.threshold_binarize(
                    x[:, var.binary_indices], thresh
                )

            # Store rounded result
            output[var.key] = x
            offset += n
        return output


class StochasticDynamicThresholdRounding(DynamicThresholdRounding):
    """
    Stochastic dynamic threshold rounding with Gumbel-Softmax noise.

    Same as DynamicThresholdRounding but uses GumbelThresholdBinarize
    for stochastic training and deterministic evaluation.

    Args:
        vars: Variable or list of variables with type metadata.
        param_keys: List of parameter keys to read from data dict.
        net: Network mapping [params, vars] to per-variable outputs.
        continuous_update: Whether to update continuous variables (default: False).
        temperature: Gumbel-Softmax temperature (default: 1.0).
        name: Module name.
    """

    def __init__(self, vars, param_keys, net,
                 continuous_update=False, temperature=1.0,
                 name="stochastic_dynamic_threshold_rounding"):
        super().__init__(vars, param_keys, net,
                         continuous_update=continuous_update,
                         name=name)
        # Replace sigmoid-smoothed binarization with Gumbel-Softmax version
        self.threshold_binarize = GumbelThresholdBinarize(
            temperature=temperature
        )
