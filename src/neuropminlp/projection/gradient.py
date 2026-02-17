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
                 num_steps=1000, step_size=0.01, decay=0.1,
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
        d = 1.0

        for _ in range(self.num_steps):
            # Build temp data with current x
            temp_data = {**data, self.target_key: x}

            # Round through components
            for comp in self.rounding_components:
                temp_data.update(comp(temp_data))

            # Compute constraint violation energy (mean of abs violations)
            violations = []
            for con in self.constraints:
                out = con(temp_data)
                viol_key = con.output_keys[2]
                viol = out[viol_key]
                violations.append(viol.reshape(x.shape[0], -1))
            
            # Early stop if feasible
            if not violations:
                break

            # Compute energy
            violations = torch.cat(violations, dim=-1)
            energy = torch.mean(torch.abs(violations), dim=1)

            # Check convergence
            if energy.max().item() < self.tolerance:
                break

            # Gradient step on x_relaxed
            grad = torch.autograd.grad(energy.sum(), x)[0]
            x = (x - d * self.step_size * grad).detach().requires_grad_(True)
            d = d - self.decay * d

        # Final update
        data[self.target_key] = x.detach()

        # Final round
        for comp in self.rounding_components:
            data.update(comp(data))

        return data
