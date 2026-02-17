"""
Unit tests for GradientProjection.
"""

import pytest
import torch
from torch import nn
from types import SimpleNamespace

from neuropminlp.projection.gradient import GradientProjection


@pytest.fixture(autouse=True)
def seed():
    """Set random seed for reproducibility."""
    torch.manual_seed(42)


class MockRounding(nn.Module):
    """Mock rounding: round to nearest integer."""

    def __init__(self, input_key="x_rel", output_key="x"):
        super().__init__()
        self.input_key = input_key
        self.output_key = output_key

    def forward(self, data):
        x_rel = data[self.input_key]
        rounded = x_rel.round()
        # Use Straight-Through Estimator (STE) to allow gradients to flow back to x_rel
        return {self.output_key: (rounded - x_rel).detach() + x_rel}


class MockConstraint(nn.Module):
    """
    Mock constraint: x <= upper_bound.

    output_keys[2] is the violation tensor (matching nm.Constraint format).
    Violation = relu(x - upper_bound).
    """

    def __init__(self, upper_bound, name="mock", input_key="x"):
        super().__init__()
        self.upper_bound = upper_bound
        self.input_key = input_key
        self.output_keys = [
            f"con_{name}",
            f"con_{name}_loss",
            f"con_{name}_viol",
        ]

    def forward(self, data):
        x = data[self.input_key]
        viol = torch.relu(x - self.upper_bound)
        penalty = viol.reshape(x.shape[0], -1).sum(dim=1).mean()
        return {
            self.output_keys[0]: penalty,
            self.output_keys[1]: penalty,
            self.output_keys[2]: viol,
        }


class MockLowerBoundConstraint(nn.Module):
    """Mock constraint: x >= lower_bound."""

    def __init__(self, lower_bound, name="mock_lb", input_key="x"):
        super().__init__()
        self.lower_bound = lower_bound
        self.input_key = input_key
        self.output_keys = [f"con_{name}", f"con_{name}_loss", f"con_{name}_viol"]

    def forward(self, data):
        x = data[self.input_key]
        # Violation if x < lower_bound => relu(lower - x)
        viol = torch.relu(self.lower_bound - x)
        penalty = viol.reshape(x.shape[0], -1).sum(dim=1).mean()
        return {
            self.output_keys[0]: penalty,
            self.output_keys[1]: penalty,
            self.output_keys[2]: viol,
        }


class TestGradientProjection:
    """Tests for GradientProjection."""

    def test_returns_dict(self):
        """Should return a dictionary."""
        proj = GradientProjection(
            rounding_components=[MockRounding()],
            constraints=[MockConstraint(upper_bound=5.0)],
            target_key="x_rel",
            num_steps=1,
        )
        data = {"x_rel": torch.tensor([[3.0, 4.0]])}
        result = proj(data)
        assert isinstance(result, dict)
        assert "x" in result
        assert "x_rel" in result

    def test_feasible_input_unchanged(self):
        """Already feasible input should not change."""
        proj = GradientProjection(
            rounding_components=[MockRounding()],
            constraints=[MockConstraint(upper_bound=10.0)],
            target_key="x_rel",
            num_steps=100,
            step_size=0.1,
        )
        data = {"x_rel": torch.tensor([[1.2, 2.8]])}
        result = proj(data)
        # Rounded: [1.0, 3.0], both <= 10, so feasible
        assert torch.allclose(result["x"], torch.tensor([[1.0, 3.0]]))

    def test_infeasible_input_projected(self):
        """Infeasible input should be projected towards feasibility."""
        proj = GradientProjection(
            rounding_components=[MockRounding()],
            constraints=[MockConstraint(upper_bound=3.0)],
            target_key="x_rel",
            num_steps=200,
            step_size=0.1,
            decay=0.99,
        )
        # x_rel = [1.5, 5.5] -> rounds to [2, 6], violates x <= 3 at index 1
        data = {"x_rel": torch.tensor([[1.5, 5.5]])}
        result = proj(data)
        # After projection, all rounded values should be <= 3
        assert (result["x"] <= 3.0 + 0.1).all()

    def test_early_stop_on_tolerance(self):
        """Should stop early when violation < tolerance."""
        proj = GradientProjection(
            rounding_components=[MockRounding()],
            constraints=[MockConstraint(upper_bound=10.0)],
            target_key="x_rel",
            num_steps=1000,
            tolerance=1e-6,
        )
        # Already feasible -> should stop immediately
        data = {"x_rel": torch.tensor([[1.0, 2.0]])}
        result = proj(data)
        assert "x" in result

    def test_step_size_decay(self):
        """Step size should decay each iteration."""
        proj = GradientProjection(
            rounding_components=[MockRounding()],
            constraints=[MockConstraint(upper_bound=3.0)],
            target_key="x_rel",
            num_steps=10,
            step_size=1.0,
            decay=0.5,
        )
        data = {"x_rel": torch.tensor([[1.5, 5.5]])}
        # Should not error; decay reduces step each iteration
        result = proj(data)
        assert "x" in result

    def test_multiple_constraints(self):
        """Should handle multiple constraints."""
        con1 = MockConstraint(upper_bound=3.0, name="upper")
        con2 = MockConstraint(upper_bound=5.0, name="upper2")
        proj = GradientProjection(
            rounding_components=[MockRounding()],
            constraints=[con1, con2],
            target_key="x_rel",
            num_steps=100,
            step_size=0.1,
        )
        data = {"x_rel": torch.tensor([[1.5, 4.5]])}
        result = proj(data)
        # Tighter constraint is x <= 3
        assert (result["x"] <= 3.0 + 0.1).all()

    def test_batch_dimension(self):
        """Should work with batch size > 1."""
        proj = GradientProjection(
            rounding_components=[MockRounding()],
            constraints=[MockConstraint(upper_bound=3.0)],
            target_key="x_rel",
            num_steps=100,
            step_size=0.1,
        )
        data = {"x_rel": torch.tensor([[1.5, 5.5], [2.0, 4.0]])}
        result = proj(data)
        assert result["x"].shape == (2, 2)

    def test_preserves_other_keys(self):
        """Should preserve other keys in data dict."""
        proj = GradientProjection(
            rounding_components=[MockRounding()],
            constraints=[MockConstraint(upper_bound=10.0)],
            target_key="x_rel",
            num_steps=1,
        )
        data = {"x_rel": torch.tensor([[1.0, 2.0]]), "params": torch.tensor([[0.5]])}
        result = proj(data)
        assert "params" in result
        assert torch.equal(result["params"], torch.tensor([[0.5]]))

    def test_final_output_is_rounded(self):
        """Final output should be rounded (integer values)."""
        proj = GradientProjection(
            rounding_components=[MockRounding()],
            constraints=[MockConstraint(upper_bound=10.0)],
            target_key="x_rel",
            num_steps=1,
        )
        data = {"x_rel": torch.tensor([[1.3, 2.7]])}
        result = proj(data)
        # MockRounding rounds to nearest integer
        assert torch.allclose(result["x"], result["x"].round())

    def test_custom_target_key(self):
        """Should work with custom target keys."""
        # Setup: y_rel -> y, constraint on y
        proj = GradientProjection(
            rounding_components=[MockRounding(input_key="y_rel", output_key="y")],
            constraints=[MockConstraint(upper_bound=3.0, input_key="y")],
            target_key="y_rel",
            num_steps=50,
            step_size=0.1,
        )
        data = {"y_rel": torch.tensor([[5.5]])}
        result = proj(data)
        
        assert "y" in result
        # Should be projected to <= 3
        assert (result["y"] <= 3.1).all()

    def test_conflicting_constraints(self):
        """Should stabilize when constraints conflict (infeasible region)."""
        # x <= 3 AND x >= 4. Gap is [3, 4].
        # Inside the gap, gradients cancel out or are zero depending on formulation.
        # It should simply stop somewhere reasonable without crashing.
        proj = GradientProjection(
            rounding_components=[MockRounding()],
            constraints=[
                MockConstraint(upper_bound=3.0, name="upper"),
                MockLowerBoundConstraint(lower_bound=4.0, name="lower")
            ],
            target_key="x_rel",
            num_steps=100,
            step_size=0.1,
        )
        data = {"x_rel": torch.tensor([[10.0]])} # Start far above
        result = proj(data)
        # It should move down towards 4.0 and stop around there
        assert result["x"].item() <= 4.5 

    def test_no_constraints(self):
        """Should just perform rounding if no constraints provided."""
        proj = GradientProjection(
            rounding_components=[MockRounding()],
            constraints=[],
            target_key="x_rel",
        )
        data = {"x_rel": torch.tensor([[1.6]])}
        result = proj(data)
        # Just rounds 1.6 -> 2.0
        assert torch.allclose(result["x"], torch.tensor([[2.0]]))
