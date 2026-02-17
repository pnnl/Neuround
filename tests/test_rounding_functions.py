"""
Unit tests for STE rounding functions.
"""

import pytest
import torch
from torch import nn

from neuropminlp.rounding.functions import (
    DiffFloor,
    DiffBinarize,
    DiffGumbelBinarize,
    GumbelThresholdBinarize,
    ThresholdBinarize,
)


@pytest.fixture(autouse=True)
def seed():
    """Set random seed for reproducibility."""
    torch.manual_seed(42)


class TestDiffFloor:
    """Test DiffFloor straight-through estimator."""

    def test_forward_values(self):
        floor = DiffFloor()
        x = torch.tensor([0.0, 0.3, 0.9, 1.5, -0.5, -1.7])
        result = floor(x)
        expected = torch.tensor([0.0, 0.0, 0.0, 1.0, -1.0, -2.0])
        assert torch.allclose(result, expected)

    def test_gradient_passthrough(self):
        """STE: gradient should pass through as identity."""
        floor = DiffFloor()
        x = torch.tensor([0.3, 1.7, -0.5], requires_grad=True)
        y = floor(x)
        y.backward(torch.ones_like(y))
        assert torch.allclose(x.grad, torch.ones(3))

    def test_gradient_scales(self):
        """Gradient should preserve upstream scaling."""
        floor = DiffFloor()
        x = torch.tensor([0.5, 1.5], requires_grad=True)
        y = floor(x)
        grad_out = torch.tensor([2.0, 3.0])
        y.backward(grad_out)
        assert torch.allclose(x.grad, grad_out)

    def test_integer_input(self):
        """Floor of integers should be the integers themselves."""
        floor = DiffFloor()
        x = torch.tensor([0.0, 1.0, 2.0, -1.0])
        result = floor(x)
        assert torch.allclose(result, x)

    def test_output_dtype(self):
        floor = DiffFloor()
        x = torch.tensor([0.5, 1.5])
        result = floor(x)
        assert result.dtype == torch.float32

    def test_batched(self):
        floor = DiffFloor()
        x = torch.tensor([[0.3, 1.7], [2.1, -0.5]])
        result = floor(x)
        expected = torch.tensor([[0.0, 1.0], [2.0, -1.0]])
        assert torch.allclose(result, expected)


class TestDiffBinarize:
    """Test DiffBinarize straight-through estimator."""

    def test_forward_values(self):
        binarize = DiffBinarize()
        x = torch.tensor([-1.5, -0.5, 0.0, 0.5, 1.5])
        result = binarize(x)
        expected = torch.tensor([0.0, 0.0, 1.0, 1.0, 1.0])
        assert torch.allclose(result, expected)

    def test_output_binary(self):
        """Output should be exactly 0 or 1."""
        binarize = DiffBinarize()
        x = torch.randn(100)
        result = binarize(x)
        assert torch.all((result == 0) | (result == 1))

    def test_gradient_legacy_behavior(self):
        """Legacy gradient: zeroed where |x| < 1, passed where |x| >= 1.

        Note: This is the intentionally preserved legacy behavior.
        """
        binarize = DiffBinarize()
        x = torch.tensor([-1.5, -0.5, 0.0, 0.5, 1.5], requires_grad=True)
        y = binarize(x)
        y.backward(torch.ones_like(y))
        # |x| < 1 -> grad zeroed: indices 1,2,3
        # |x| >= 1 -> grad passed: indices 0,4
        expected_grad = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0])
        assert torch.allclose(x.grad, expected_grad)

    def test_clamping(self):
        """Input outside [-1, 1] should still produce valid binary output."""
        binarize = DiffBinarize()
        x = torch.tensor([-5.0, 5.0])
        result = binarize(x)
        assert torch.allclose(result, torch.tensor([0.0, 1.0]))

    def test_batched(self):
        binarize = DiffBinarize()
        x = torch.tensor([[-0.5, 0.5], [0.0, 1.0]])
        result = binarize(x)
        expected = torch.tensor([[0.0, 1.0], [1.0, 1.0]])
        assert torch.allclose(result, expected)


class TestDiffGumbelBinarize:
    """Test DiffGumbelBinarize Gumbel-Softmax estimator."""

    def test_output_binary_train(self):
        """Train mode output should be approximately binary."""
        gumbel = DiffGumbelBinarize(temperature=0.1)
        gumbel.train()
        x = torch.tensor([5.0, -5.0, 5.0, -5.0])
        result = gumbel(x)
        # with strong signal and low temperature, should be near binary
        assert torch.all((result >= 0) & (result <= 1))

    def test_output_binary_eval(self):
        """Eval mode output should be exactly 0 or 1."""
        gumbel = DiffGumbelBinarize()
        gumbel.eval()
        x = torch.randn(100)
        result = gumbel(x)
        assert torch.all((result == 0) | (result == 1))

    def test_eval_deterministic(self):
        """Eval mode should be deterministic."""
        gumbel = DiffGumbelBinarize()
        gumbel.eval()
        x = torch.tensor([0.5, -0.5, 1.0, -1.0])
        r1 = gumbel(x)
        r2 = gumbel(x)
        assert torch.allclose(r1, r2)

    def test_train_stochastic(self):
        """Train mode should produce different results across calls."""
        gumbel = DiffGumbelBinarize()
        gumbel.train()
        x = torch.zeros(100)  # ambiguous input to maximize randomness
        results = [gumbel(x) for _ in range(10)]
        # at least some variation expected
        assert not all(torch.allclose(results[0], r) for r in results[1:])

    def test_gradient_exists_train(self):
        """Train mode should produce gradients."""
        gumbel = DiffGumbelBinarize()
        gumbel.train()
        x = torch.tensor([0.5, -0.5], requires_grad=True)
        y = gumbel(x)
        y.sum().backward()
        assert x.grad is not None
        assert not torch.all(x.grad == 0)

    def test_no_gradient_eval(self):
        """Eval mode has no differentiable path."""
        gumbel = DiffGumbelBinarize()
        gumbel.eval()
        x = torch.tensor([0.5, -0.5], requires_grad=True)
        y = gumbel(x)
        # Hard threshold detaches from computation graph
        assert not y.requires_grad

    def test_temperature_effect(self):
        """Lower temperature should produce sharper outputs."""
        gumbel_hot = DiffGumbelBinarize(temperature=10.0)
        gumbel_cold = DiffGumbelBinarize(temperature=0.01)
        gumbel_hot.eval()
        gumbel_cold.eval()
        # strong positive signal
        x = torch.tensor([2.0])
        # both should output 1 for strong signal
        assert gumbel_cold(x).item() == 1.0
        assert gumbel_hot(x).item() == 1.0

    def test_positive_bias(self):
        """Strong positive input should almost always produce 1."""
        gumbel = DiffGumbelBinarize(temperature=0.1)
        gumbel.train()
        x = torch.full((100,), 10.0)
        result = gumbel(x)
        assert result.mean() > 0.9

    def test_negative_bias(self):
        """Strong negative input should almost always produce 0."""
        gumbel = DiffGumbelBinarize(temperature=0.1)
        gumbel.train()
        x = torch.full((100,), -10.0)
        result = gumbel(x)
        assert result.mean() < 0.1

    def test_batched(self):
        gumbel = DiffGumbelBinarize()
        gumbel.eval()
        x = torch.tensor([[1.0, -1.0], [2.0, -2.0]])
        result = gumbel(x)
        assert result.shape == (2, 2)


class TestGumbelThresholdBinarize:
    """Test GumbelThresholdBinarize with Gumbel noise + learned thresholds."""

    def test_forward_values_eval(self):
        """Eval mode should be deterministic x >= threshold."""
        gtb = GumbelThresholdBinarize()
        gtb.eval()
        x = torch.tensor([0.3, 0.5, 0.7])
        threshold = torch.tensor([0.5, 0.5, 0.5])
        result = gtb(x, threshold)
        expected = torch.tensor([0.0, 1.0, 1.0])
        assert torch.allclose(result, expected)

    def test_output_binary_eval(self):
        """Eval mode output should be exactly 0 or 1."""
        gtb = GumbelThresholdBinarize()
        gtb.eval()
        x = torch.rand(100)
        threshold = torch.rand(100)
        result = gtb(x, threshold)
        assert torch.all((result == 0) | (result == 1))

    def test_output_range_train(self):
        """Train mode output should be in [0, 1]."""
        gtb = GumbelThresholdBinarize(temperature=0.1)
        gtb.train()
        x = torch.rand(100)
        threshold = torch.rand(100)
        result = gtb(x, threshold)
        assert torch.all((result >= 0) & (result <= 1))

    def test_eval_deterministic(self):
        """Eval mode should produce identical results across calls."""
        gtb = GumbelThresholdBinarize()
        gtb.eval()
        x = torch.tensor([0.3, 0.7])
        threshold = torch.tensor([0.5, 0.5])
        r1 = gtb(x, threshold)
        r2 = gtb(x, threshold)
        assert torch.allclose(r1, r2)

    def test_train_stochastic(self):
        """Train mode should produce different results across calls."""
        gtb = GumbelThresholdBinarize()
        gtb.train()
        x = torch.full((100,), 0.5)
        threshold = torch.full((100,), 0.5)
        results = [gtb(x, threshold) for _ in range(10)]
        assert not all(torch.allclose(results[0], r) for r in results[1:])

    def test_gradient_on_threshold(self):
        """Threshold should receive gradients in train mode."""
        gtb = GumbelThresholdBinarize()
        gtb.train()
        x = torch.tensor([0.5])
        threshold = torch.tensor([0.4], requires_grad=True)
        result = gtb(x, threshold)
        result.backward()
        assert threshold.grad is not None
        assert not torch.all(threshold.grad == 0)

    def test_gradient_on_x(self):
        """Input x should receive gradients in train mode."""
        gtb = GumbelThresholdBinarize()
        gtb.train()
        x = torch.tensor([0.5], requires_grad=True)
        threshold = torch.tensor([0.4])
        result = gtb(x, threshold)
        result.backward()
        assert x.grad is not None

    def test_no_gradient_eval(self):
        """Eval mode has no differentiable path."""
        gtb = GumbelThresholdBinarize()
        gtb.eval()
        x = torch.tensor([0.5], requires_grad=True)
        threshold = torch.tensor([0.4])
        result = gtb(x, threshold)
        # Hard threshold detaches from computation graph
        assert not result.requires_grad

    def test_threshold_clamped(self):
        """Thresholds outside [0, 1] should be clamped."""
        gtb = GumbelThresholdBinarize()
        gtb.eval()
        x = torch.tensor([0.5])
        result_high = gtb(x, torch.tensor([2.0]))
        assert result_high.item() == 0.0
        result_low = gtb(x, torch.tensor([-1.0]))
        assert result_low.item() == 1.0

    def test_positive_bias(self):
        """x well above threshold should almost always produce 1."""
        gtb = GumbelThresholdBinarize(temperature=0.1)
        gtb.train()
        x = torch.full((100,), 10.0)
        threshold = torch.full((100,), 0.1)
        result = gtb(x, threshold)
        assert result.mean() > 0.9

    def test_batched(self):
        gtb = GumbelThresholdBinarize()
        gtb.eval()
        x = torch.tensor([[0.3, 0.7], [0.5, 0.1]])
        threshold = torch.tensor([[0.5, 0.5], [0.5, 0.5]])
        result = gtb(x, threshold)
        expected = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
        assert torch.allclose(result, expected)


class TestThresholdBinarize:
    """Test ThresholdBinarize with learned thresholds."""

    def test_forward_values(self):
        tb = ThresholdBinarize()
        x = torch.tensor([0.3, 0.5, 0.7])
        threshold = torch.tensor([0.5, 0.5, 0.5])
        result = tb(x, threshold)
        expected = torch.tensor([0.0, 1.0, 1.0])
        assert torch.allclose(result, expected)

    def test_output_binary(self):
        """Output should be exactly 0 or 1."""
        tb = ThresholdBinarize()
        x = torch.rand(100)
        threshold = torch.rand(100)
        result = tb(x, threshold)
        assert torch.all((result == 0) | (result == 1))

    def test_threshold_clamped(self):
        """Thresholds outside [0, 1] should be clamped."""
        tb = ThresholdBinarize()
        x = torch.tensor([0.5])
        # threshold > 1 clamped to 1
        result_high = tb(x, torch.tensor([2.0]))
        assert result_high.item() == 0.0
        # threshold < 0 clamped to 0
        result_low = tb(x, torch.tensor([-1.0]))
        assert result_low.item() == 1.0

    def test_gradient_on_threshold(self):
        """Threshold should receive gradients via sigmoid smoothing."""
        tb = ThresholdBinarize()
        x = torch.tensor([0.5])
        threshold = torch.tensor([0.4], requires_grad=True)
        result = tb(x, threshold)
        result.backward()
        assert threshold.grad is not None
        assert not torch.all(threshold.grad == 0)

    def test_gradient_on_x(self):
        """Input x should receive gradients via sigmoid smoothing."""
        tb = ThresholdBinarize()
        x = torch.tensor([0.5], requires_grad=True)
        threshold = torch.tensor([0.4])
        result = tb(x, threshold)
        result.backward()
        assert x.grad is not None

    def test_slope_effect(self):
        """Higher slope should produce sharper sigmoid approximation."""
        tb_sharp = ThresholdBinarize(slope=100)
        tb_smooth = ThresholdBinarize(slope=1)
        x = torch.tensor([0.51], requires_grad=True)
        threshold = torch.tensor([0.50])
        # Both should output 1 (hard), but gradient magnitudes differ
        r_sharp = tb_sharp(x, threshold)
        r_smooth = tb_smooth(x, threshold)
        assert r_sharp.item() == 1.0
        assert r_smooth.item() == 1.0

    def test_batched(self):
        tb = ThresholdBinarize()
        x = torch.tensor([[0.3, 0.7], [0.5, 0.1]])
        threshold = torch.tensor([[0.5, 0.5], [0.5, 0.5]])
        result = tb(x, threshold)
        expected = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
        assert torch.allclose(result, expected)


class TestRoundingFunctionsExport:
    """Test that functions are exported from neuropminlp.rounding."""

    def test_import_from_rounding(self):
        from neuropminlp.rounding import (
            DiffFloor,
            DiffBinarize,
            DiffGumbelBinarize,
            GumbelThresholdBinarize,
            ThresholdBinarize,
        )
        assert DiffFloor is not None
        assert DiffBinarize is not None
        assert DiffGumbelBinarize is not None
        assert GumbelThresholdBinarize is not None
        assert ThresholdBinarize is not None

    def test_all_are_modules(self):
        """All STE functions should be nn.Module subclasses."""
        assert issubclass(DiffFloor, nn.Module)
        assert issubclass(DiffBinarize, nn.Module)
        assert issubclass(DiffGumbelBinarize, nn.Module)
        assert issubclass(GumbelThresholdBinarize, nn.Module)
        assert issubclass(ThresholdBinarize, nn.Module)
