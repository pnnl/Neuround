"""
Base class for rounding nodes.
"""

from abc import ABC, abstractmethod

from neuromancer.system import Node


class RoundingNode(Node, ABC):
    """
    Base class for differentiable rounding nodes.

    Inherits from neuromancer Node for compatibility with Problem.step().
    forward() takes a data dict and returns only the output dict
    {output_key: value}, which Problem merges via dict unpacking.

    Accepts single variable or list of variables. Type metadata
    is auto-extracted from each variable object.

    Args:
        vars: Variable or list of variables (created by neuropminlp.variable).
        name: Module name.
    """

    def __init__(self, vars, name="rounding"):
        # Normalize single variable to list
        if not isinstance(vars, (list, tuple)):
            vars = [vars]

        # Initialize Node with input/output keys
        input_keys = [v.relaxed.key for v in vars]
        output_keys = [v.key for v in vars]
        super().__init__(
            callable=None,
            input_keys=input_keys,
            output_keys=output_keys,
            name=name,
        )

        # Variable metadata for rounding logic
        self.vars = vars
        # Total variable count across all variables (for MLP output sizing)
        self.num_vars = sum(v.num_vars for v in vars)

    @abstractmethod
    def forward(self, data):
        """
        Round continuous variables in data dict.

        Args:
            data: Dictionary containing variable tensors.

        Returns:
            Dictionary with only rounded output variables
            (e.g., {"x": rounded_x, "y": rounded_y}).
        """
        pass
