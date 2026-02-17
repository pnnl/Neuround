"""
Gradient-based feasibility projection.
"""

import torch


class GradientProjection:
    """
    Gradient-based feasibility projection.

    This preserves integer feasibility because x_continuous is
    always re-rounded through the rounding layer.

    Args:
        rounding_components: List of rounding modules (applied sequentially).
        constraints: List of nm.Constraint objects (same ones used in PenaltyLoss).
        target_key: Dictionary key for x_continuous (e.g., "x_rel").
        num_steps: Maximum projection iterations.
        step_size: Initial step size for gradient descent.
        decay: Step size decay factor per iteration.
        tolerance: Stop if max violation < tolerance.
    """

    def __init__(self, rounding_components, constraints, target_key,
                 num_steps=1000, step_size=0.01, decay=1.0,
                 tolerance=1e-6):
        self.rounding_components = rounding_components
        self.constraints = constraints
        self.target_key = target_key
        self.num_steps = num_steps
        self.step_size = step_size
        self.decay = decay
        self.tolerance = tolerance

    def __call__(self, data):
        """
        Project x_continuous towards feasibility.

        Args:
            data: Dictionary containing target_key and parameters.

        Returns:
            Updated dictionary with projected x_continuous and
            final rounded solution.
        """
        x = data[self.target_key].clone().requires_grad_(True)
        step = self.step_size

        for _ in range(self.num_steps):
            # Build temp data with current x
            temp_data = {**data, self.target_key: x}

            # Round through components
            for comp in self.rounding_components:
                temp_data.update(comp(temp_data))

            # Compute total violation from nm.Constraint objects
            total_viol = torch.zeros(x.shape[0], device=x.device)
            for con in self.constraints:
                out = con(temp_data)
                viol_key = con.output_keys[2]
                viol = out[viol_key]
                total_viol = total_viol + viol.reshape(x.shape[0], -1).sum(dim=1)

            # Check convergence
            if total_viol.max().item() < self.tolerance:
                break

            # Gradient step on x_relaxed
            grad = torch.autograd.grad(total_viol.sum(), x)[0]
            x = (x - step * grad).detach().requires_grad_(True)
            step *= self.decay

        # Final update
        data[self.target_key] = x.detach()

        # Final round
        for comp in self.rounding_components:
            data.update(comp(data))

        return data
