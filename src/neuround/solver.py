"""
LearnableSolver: convenience wrapper for end-to-end learnable MINLP solver.
"""

import functools

import torch

from neuromancer.problem import Problem

from neuround.projection.gradient import GradientProjection


def fast(problem, device="cuda", compile=True):
    """
    Decorate a Problem with torch.compile + AMP autocast.

    Applied implicitly inside the package — users never call this
    directly.  Works transparently with neuromancer Trainer.

    Args:
        problem: neuromancer Problem instance.
        device: Target device (AMP is enabled only for "cuda").

    Returns:
        The (possibly compiled) problem with autocast-wrapped forward.
    """
    use_amp = "cuda" in str(device)

    # Wrap forward with autocast first (compile will trace through it)
    _orig_forward = problem.forward

    @functools.wraps(_orig_forward)
    def _amp_forward(*args, **kwargs):
        with torch.amp.autocast("cuda", enabled=use_amp):
            return _orig_forward(*args, **kwargs)

    problem.forward = _amp_forward

    # torch.compile for graph-level fusion (graceful fallback)
    if compile:
        try:
            problem = torch.compile(problem)
        except Exception:
            pass

    return problem


class LearnableSolver:
    """
    Convenience wrapper composing smap + rounding + loss into a
    neuromancer Problem with optional GradientProjection for inference.

    Args:
        smap_node: Node wrapping the solution mapper MLP.
        rounding_node: RoundingNode instance.
        loss: PenaltyLoss instance (objectives + constraints).
        projection_steps: Max projection iterations (default 1000).
        projection_step_size: Step size for gradient projection.
        projection_decay: Decay rate for projection step size.
    """

    def __init__(self, smap_node, rounding_node, loss,
                 projection_steps=1000,
                 projection_step_size=0.01,
                 projection_decay=1.0):
        # Store solution mapping node
        self.smap_node = smap_node
        # Store rounding node
        self.rounding_node = rounding_node
        # Store loss
        self.loss = loss
        # Store projection config
        self.projection_steps = projection_steps
        self.projection_step_size = projection_step_size
        self.projection_decay = projection_decay

        # Validate key alignment between smap outputs and rounding inputs
        self._validate_key_alignment()
        # Validate dimension alignment between smap outputs and rounding indices
        self._validate_dimension_alignment()
        # Build neuromancer Problem with smap and rounding nodes
        self._build_problem()
        # Build optional GradientProjection for inference
        self._build_projection(list(self.loss.constraints))

    def _validate_key_alignment(self):
        """Check smap output keys cover rounding input keys."""
        # Collect smap output keys
        smap_outputs = set(self.smap_node.output_keys)
        # Collect relaxed keys
        rounding_rel_keys = {v.relaxed.key for v in self.rounding_node.vars}
        # Check all rounding relaxed keys are in smap outputs
        missing = rounding_rel_keys - smap_outputs
        if missing:
            raise ValueError(
                f"Key mismatch: rounding layer expects relaxed keys "
                f"{missing} but smap_node outputs "
                f"{self.smap_node.output_keys}"
            )

    def _validate_dimension_alignment(self):
        """Check rounding indices do not exceed smap output dimension."""
        # Get smap output dimension
        out_dim = getattr(self.smap_node.callable, 'out_features', None)
        # Fallback to 'outsize' attribute if 'out_features' is not available
        if out_dim is None:
            out_dim = getattr(self.smap_node.callable, 'outsize', None)
        # Check against rounding variable indices
        if out_dim is not None:
            # Collect rounding indices
            for var in self.rounding_node.vars:
                # Check integer and binary indices against smap output dimension
                all_idx = var.integer_indices + var.binary_indices
                if all_idx and max(all_idx) >= out_dim:
                    raise ValueError(
                        f"Variable '{var.key}': rounding index "
                        f"{max(all_idx)} >= smap output dim {out_dim}"
                    )

    def _build_problem(self):
        """Assemble neuromancer Problem from smap and rounding nodes."""
        self.problem = Problem(
            nodes=[self.smap_node, self.rounding_node],
            loss=self.loss,
        )

    def _build_projection(self, constraints):
        """Build GradientProjection for inference (skipped without constraints)."""
        # Projection is only meaningful if constraints are provided
        if self.projection_steps > 0 and len(constraints) > 0:
            # Collect relaxed keys from all rounding variables
            target_keys = [v.relaxed.key for v in self.rounding_node.vars]
            self.projection = GradientProjection(
                rounding_components=[self.rounding_node],
                constraints=constraints,
                target_keys=target_keys,
                num_steps=self.projection_steps,
                step_size=self.projection_step_size,
                decay=self.projection_decay,
            )
        else:
            # No projection if no constraints or zero steps
            self.projection = None

    def train(self, loader_train, loader_val, optimizer,
              epochs=200, patience=20, warmup=20,
              device="cpu", compile=True):
        """
        Train the solver with AMP autocast and optional torch.compile.

        Args:
            loader_train: Training DataLoader.
            loader_val: Validation DataLoader.
            optimizer: Optimizer instance (e.g. AdamW).
            epochs: Max training epochs.
            patience: Early stopping patience.
            warmup: Warmup epochs before early stopping.
            device: Training device.
        """
        from neuromancer.trainer import Trainer

        self.problem.to(device)
        decorated = fast(self.problem, device=device, compile=compile)
        trainer = Trainer(
            decorated, loader_train, loader_val,
            optimizer=optimizer, epochs=epochs,
            patience=patience, warmup=warmup,
            device=device,
        )
        best_model = trainer.train()
        # Strip "_orig_mod." prefix added by torch.compile
        clean = {k.removeprefix("_orig_mod."): v for k, v in best_model.items()}
        self.problem.load_state_dict(clean)

    def predict(self, data):
        """
        Inference: params -> projected rounded solution.

        Without projection: smap -> rounding (no_grad).
        With projection: smap (no_grad) -> projection (grad enabled).

        Args:
            data: Dictionary with parameter tensors.

        Returns:
            Updated dictionary with solution.
        """
        # Ensure model is in eval mode
        self.problem.eval()

        # Get relaxed solution from smap
        with torch.no_grad():
            data.update(self.smap_node(data))

        # Apply projection (handles rounding internally)
        if self.projection is not None:
            data = self.projection(data)
        # Apply rounding only
        else:
            with torch.no_grad():
                data.update(self.rounding_node(data))

        return data
