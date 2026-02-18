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
        target_keys: List of dictionary keys for relaxed variables
            (e.g., ["x_rel", "y_rel"]).
        num_steps: Maximum projection iterations.
        step_size: Initial step size for gradient descent.
        decay: Step size decay factor per iteration.
        tolerance: Stop if max violation < tolerance.
    """

    def __init__(self, rounding_components, constraints, target_keys,
                 num_steps=1000, step_size=0.01, decay=1.0,
                 tolerance=1e-6):
        self.rounding_components = rounding_components
        self.constraints = constraints
        self.target_keys = target_keys
        self.num_steps = num_steps
        self.step_size = step_size
        self.decay = decay
        self.tolerance = tolerance

    def __call__(self, data):
        """
        Project relaxed variables towards feasibility.

        Args:
            data: Dictionary containing target keys and parameters.

        Returns:
            Updated dictionary with projected relaxed variables
            and final rounded solution.
        """
        # Clone and enable grad for all target variables
        xs = {k: data[k].clone().requires_grad_(True) for k in self.target_keys}
        batch_size = next(iter(xs.values())).shape[0]
        device = next(iter(xs.values())).device
        d = 1.0

        # Create a shallow copy once to avoid repeated dict allocation overhead
        temp_data = data.copy()

        for i in range(self.num_steps):
            # Update temp data with current xs (in-place update is faster)
            temp_data.update(xs)

            # Round through components
            for comp in self.rounding_components:
                temp_data.update(comp(temp_data))

            # Compute total violation from constraints
            total_viol = torch.zeros(batch_size, device=device)
            for con in self.constraints:
                out = con(temp_data)
                viol_key = con.output_keys[2]
                viol = out[viol_key]
                total_viol = total_viol + viol.reshape(batch_size, -1).sum(dim=1)

            # Check convergence
            if total_viol.max().item() < self.tolerance:
                break

            # Gradient step on all target variables
            grads = torch.autograd.grad(total_viol.sum(), list(xs.values()), allow_unused=True)
            xs = {
                k: (xs[k] - d * self.step_size * (g if g is not None else 0.0)).detach().requires_grad_(True)
                for k, g in zip(self.target_keys, grads)
            }
            d = self.decay * d

        # Final update
        for k in self.target_keys:
            data[k] = xs[k].detach()

        # Final round
        for comp in self.rounding_components:
            data.update(comp(data))

        return data
