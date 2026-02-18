"""
Unit tests for rounding layers.
"""

import pytest
import torch
from types import SimpleNamespace
from torch import nn

from neuround.blocks import MLPBnDrop
from neuromancer.system import Node
from neuround.rounding.base import RoundingNode
from neuround.rounding.ste import STERounding
from neuround.rounding.threshold import (
    DynamicThresholdRounding,
    StochasticDynamicThresholdRounding,
)
from neuround.rounding.selection import (
    AdaptiveSelectionRounding,
    StochasticAdaptiveSelectionRounding,
)


@pytest.fixture(autouse=True)
def seed():
    """Set random seed for reproducibility."""
    torch.manual_seed(42)


def _make_var(key, num_vars, integer_indices=None, binary_indices=None):
    """Create a mock variable with type metadata for testing."""
    integer_indices = integer_indices or []
    binary_indices = binary_indices or []
    continuous_indices = [
        i for i in range(num_vars)
        if i not in integer_indices and i not in binary_indices
    ]
    relaxed = SimpleNamespace(key=key + "_rel")
    return SimpleNamespace(
        key=key,
        relaxed=relaxed,
        num_vars=num_vars,
        integer_indices=integer_indices,
        binary_indices=binary_indices,
        continuous_indices=continuous_indices,
    )


def _make_net(num_params, num_vars):
    """Create a small MLP for testing: [params, x_all] -> num_vars."""
    return MLPBnDrop(
        insize=num_params + num_vars,
        outsize=num_vars,
        hsizes=[16],
    )


# ── Fixtures ─────────────────────────────────────────────────────────

@pytest.fixture
def int_var():
    """Variable with all integer indices."""
    return _make_var("x", 3, integer_indices=[0, 1, 2])


@pytest.fixture
def bin_var():
    """Variable with all binary indices."""
    return _make_var("x", 3, binary_indices=[0, 1, 2])


@pytest.fixture
def mixed_var():
    """Variable with mixed types: continuous, integer, binary."""
    return _make_var("x", 5, integer_indices=[1, 3], binary_indices=[2, 4])


@pytest.fixture
def multi_vars():
    """Two variables for multi-variable tests."""
    x = _make_var("x", 3, integer_indices=[0, 1, 2])
    y = _make_var("y", 2, binary_indices=[0, 1])
    return [x, y]


# ── TestRoundingNodeBase ─────────────────────────────────────────────

class TestRoundingNodeBase:
    """Test RoundingNode base class metadata extraction."""

    def test_single_var_normalized_to_list(self, int_var):
        """Single variable should be wrapped in a list."""
        layer = STERounding(int_var)
        assert isinstance(layer.vars, list)
        assert len(layer.vars) == 1

    def test_input_keys_single(self, int_var):
        layer = STERounding(int_var)
        assert layer.input_keys == ["x_rel"]
        assert layer.output_keys == ["x"]

    def test_input_keys_multi(self, multi_vars):
        layer = STERounding(multi_vars)
        assert layer.input_keys == ["x_rel", "y_rel"]
        assert layer.output_keys == ["x", "y"]

    def test_num_vars_single(self, int_var):
        layer = STERounding(int_var)
        assert layer.num_vars == 3

    def test_num_vars_multi(self, multi_vars):
        """Total num_vars should be sum across all variables."""
        layer = STERounding(multi_vars)
        assert layer.num_vars == 5  # 3 + 2

    def test_name(self, int_var):
        layer = STERounding(int_var, name="my_rounding")
        assert layer.name == "my_rounding"

    def test_input_keys_with_params(self, int_var):
        """Learnable layers should include param_keys in input_keys."""
        net = _make_net(4, 3)
        layer = DynamicThresholdRounding(
            int_var, param_keys=["p"], net=net
        )
        assert layer.input_keys == ["p", "x_rel"]
        assert layer.output_keys == ["x"]

    def test_is_node(self, int_var):
        """RoundingNode should be a neuromancer Node."""
        layer = STERounding(int_var)
        assert isinstance(layer, Node)
        assert isinstance(layer, nn.Module)

    def test_forward_returns_output_only(self, int_var):
        """Forward should return only output keys, not full data dict."""
        layer = STERounding(int_var)
        data = {"x_rel": torch.randn(4, 3), "extra": torch.randn(4, 2)}
        result = layer(data)
        assert set(result.keys()) == {"x"}

    def test_forward_does_not_mutate_data(self, int_var):
        """Forward should not add keys to the input data dict."""
        layer = STERounding(int_var)
        data = {"x_rel": torch.randn(4, 3)}
        original_keys = set(data.keys())
        layer(data)
        assert set(data.keys()) == original_keys


# ── TestSTERounding ──────────────────────────────────────────────────

class TestSTERounding:
    """Test STERounding layer."""

    def test_integer_rounding(self, int_var):
        """Integer variables should be rounded to integers."""
        layer = STERounding(int_var)
        layer.eval()
        data = {"x_rel": torch.tensor([[1.3, 2.7, 0.1]])}
        result = layer(data)
        x = result["x"]
        assert torch.all(x == x.floor()) or torch.all(x == x.ceil())

    def test_binary_rounding(self, bin_var):
        """Binary variables should be 0 or 1."""
        layer = STERounding(bin_var)
        layer.eval()
        data = {"x_rel": torch.tensor([[0.3, 0.7, 0.5]])}
        result = layer(data)
        x = result["x"]
        assert torch.all((x == 0) | (x == 1))

    def test_mixed_types(self, mixed_var):
        """Mixed variable: continuous untouched, integers rounded, binaries 0/1."""
        layer = STERounding(mixed_var)
        layer.eval()
        x_rel = torch.tensor([[0.5, 1.7, 0.8, 3.2, 0.3]])
        data = {"x_rel": x_rel}
        result = layer(data)
        x = result["x"]
        # Continuous indices [0] should be unchanged
        assert torch.allclose(x[:, [0]], x_rel[:, [0]])
        # Binary indices [2, 4] should be 0 or 1
        assert torch.all((x[:, [2, 4]] == 0) | (x[:, [2, 4]] == 1))

    def test_output_shape(self, int_var):
        layer = STERounding(int_var)
        data = {"x_rel": torch.randn(8, 3)}
        result = layer(data)
        assert result["x"].shape == (8, 3)

    def test_multi_var(self, multi_vars):
        """Multi-variable: each variable rounded independently."""
        layer = STERounding(multi_vars)
        layer.eval()
        data = {
            "x_rel": torch.tensor([[1.3, 2.7, 0.1]]),
            "y_rel": torch.tensor([[0.8, 0.2]]),
        }
        result = layer(data)
        # x: all integer
        x = result["x"]
        assert torch.all(x == x.floor()) or torch.all(x == x.ceil())
        # y: all binary
        y = result["y"]
        assert torch.all((y == 0) | (y == 1))

    def test_gradient_flow(self, int_var):
        """Gradients should flow through STE rounding."""
        layer = STERounding(int_var)
        layer.train()
        x_rel = torch.tensor([[1.3, 2.7, 0.1]], requires_grad=True)
        data = {"x_rel": x_rel}
        result = layer(data)
        result["x"].sum().backward()
        assert x_rel.grad is not None
        assert not torch.all(x_rel.grad == 0)

    def test_no_learnable_params(self, int_var):
        """STERounding should have no learnable parameters."""
        layer = STERounding(int_var)
        params = list(layer.parameters())
        assert len(params) == 0

    def test_train_eval_consistency(self, int_var):
        """Eval mode should produce deterministic results."""
        layer = STERounding(int_var)
        layer.eval()
        data1 = {"x_rel": torch.tensor([[1.3, 2.7, 0.1]])}
        data2 = {"x_rel": torch.tensor([[1.3, 2.7, 0.1]])}
        r1 = layer(data1)["x"]
        r2 = layer(data2)["x"]
        assert torch.allclose(r1, r2)

    def test_batched(self, int_var):
        """Batch dimension should be preserved."""
        layer = STERounding(int_var)
        layer.eval()
        data = {"x_rel": torch.randn(16, 3)}
        result = layer(data)
        assert result["x"].shape == (16, 3)


# ── TestDynamicThresholdRounding ─────────────────────────────────────

class TestDynamicThresholdRounding:
    """Test DynamicThresholdRounding layer."""

    def test_output_shape(self, int_var):
        net = _make_net(4, 3)
        layer = DynamicThresholdRounding(int_var, param_keys=["p"], net=net)
        data = {"x_rel": torch.randn(8, 3), "p": torch.randn(8, 4)}
        result = layer(data)
        assert result["x"].shape == (8, 3)

    def test_integer_rounding(self, int_var):
        """Integer variables should be rounded."""
        net = _make_net(4, 3)
        layer = DynamicThresholdRounding(int_var, param_keys=["p"], net=net)
        layer.eval()
        data = {"x_rel": torch.tensor([[1.3, 2.7, 0.1]]), "p": torch.randn(1, 4)}
        result = layer(data)
        x = result["x"]
        # Each value should be an integer
        assert torch.allclose(x, x.round(), atol=0.01)

    def test_binary_rounding(self, bin_var):
        """Binary variables should be 0 or 1."""
        net = _make_net(4, 3)
        layer = DynamicThresholdRounding(bin_var, param_keys=["p"], net=net)
        layer.eval()
        data = {"x_rel": torch.tensor([[0.3, 0.7, 0.5]]), "p": torch.randn(1, 4)}
        result = layer(data)
        x = result["x"]
        assert torch.all((x == 0) | (x == 1))

    def test_gradient_flow_through_net(self, int_var):
        """Gradients should flow through network parameters."""
        net = _make_net(4, 3)
        layer = DynamicThresholdRounding(int_var, param_keys=["p"], net=net)
        layer.train()
        # Batch size >= 2 required for BatchNorm in train mode
        data = {
            "x_rel": torch.tensor([[1.3, 2.7, 0.1], [0.5, 1.2, 2.8]], requires_grad=True),
            "p": torch.randn(2, 4),
        }
        result = layer(data)
        result["x"].sum().backward()
        # Network should have gradients
        for param in layer.net.parameters():
            if param.requires_grad:
                assert param.grad is not None

    def test_has_learnable_params(self, int_var):
        """Should have learnable network parameters."""
        net = _make_net(4, 3)
        layer = DynamicThresholdRounding(int_var, param_keys=["p"], net=net)
        params = list(layer.parameters())
        assert len(params) > 0

    def test_multi_var(self, multi_vars):
        """Multi-variable with shared network."""
        net = _make_net(4, 5)  # 3 + 2 = 5 total vars
        layer = DynamicThresholdRounding(
            multi_vars, param_keys=["p"], net=net
        )
        layer.eval()
        data = {
            "x_rel": torch.tensor([[1.3, 2.7, 0.1]]),
            "y_rel": torch.tensor([[0.8, 0.2]]),
            "p": torch.randn(1, 4),
        }
        result = layer(data)
        assert result["x"].shape == (1, 3)
        assert result["y"].shape == (1, 2)
        # y: all binary -> 0 or 1
        assert torch.all((result["y"] == 0) | (result["y"] == 1))

    def test_multi_param_keys(self, int_var):
        """Multiple param_keys should be concatenated."""
        net = _make_net(6, 3)  # p1=2 + p2=4 = 6
        layer = DynamicThresholdRounding(
            int_var, param_keys=["p1", "p2"], net=net
        )
        data = {
            "x_rel": torch.randn(4, 3),
            "p1": torch.randn(4, 2),
            "p2": torch.randn(4, 4),
        }
        result = layer(data)
        assert result["x"].shape == (4, 3)

    def test_continuous_update(self, mixed_var):
        """With continuous_update, continuous variables should be adjusted."""
        net = _make_net(4, 5)
        layer = DynamicThresholdRounding(
            mixed_var, param_keys=["p"], net=net,
            continuous_update=True,
        )
        layer.eval()
        data = {
            "x_rel": torch.tensor([[0.5, 1.7, 0.8, 3.2, 0.3]]),
            "p": torch.randn(1, 4),
        }
        result = layer(data)
        assert result["x"].shape == (1, 5)

    def test_batched(self, int_var):
        net = _make_net(4, 3)
        layer = DynamicThresholdRounding(int_var, param_keys=["p"], net=net)
        data = {"x_rel": torch.randn(16, 3), "p": torch.randn(16, 4)}
        result = layer(data)
        assert result["x"].shape == (16, 3)


# ── TestStochasticDynamicThresholdRounding ───────────────────────────

class TestStochasticDynamicThresholdRounding:
    """Test StochasticDynamicThresholdRounding layer."""

    def test_inherits_dynamic_threshold(self):
        """Should be a subclass of DynamicThresholdRounding."""
        assert issubclass(
            StochasticDynamicThresholdRounding, DynamicThresholdRounding
        )

    def test_output_shape(self, int_var):
        net = _make_net(4, 3)
        layer = StochasticDynamicThresholdRounding(
            int_var, param_keys=["p"], net=net
        )
        data = {"x_rel": torch.randn(8, 3), "p": torch.randn(8, 4)}
        result = layer(data)
        assert result["x"].shape == (8, 3)

    def test_binary_rounding(self, bin_var):
        """Binary variables should be 0 or 1."""
        net = _make_net(4, 3)
        layer = StochasticDynamicThresholdRounding(
            bin_var, param_keys=["p"], net=net
        )
        layer.eval()
        data = {"x_rel": torch.tensor([[0.3, 0.7, 0.5]]), "p": torch.randn(1, 4)}
        result = layer(data)
        x = result["x"]
        assert torch.all((x == 0) | (x == 1))

    def test_train_stochastic(self, int_var):
        """Train mode should produce different results across calls."""
        net = _make_net(4, 3)
        layer = StochasticDynamicThresholdRounding(
            int_var, param_keys=["p"], net=net
        )
        layer.train()
        # Batch size >= 2 required for BatchNorm in train mode
        p = torch.randn(2, 4)
        results = []
        for _ in range(10):
            data = {"x_rel": torch.tensor([[1.5, 2.5, 0.5], [0.3, 1.8, 2.1]]), "p": p}
            results.append(layer(data)["x"])
        assert not all(torch.allclose(results[0], r) for r in results[1:])

    def test_gradient_flow(self, int_var):
        """Gradients should flow through Gumbel-Softmax path."""
        net = _make_net(4, 3)
        layer = StochasticDynamicThresholdRounding(
            int_var, param_keys=["p"], net=net
        )
        layer.train()
        # Batch size >= 2 required for BatchNorm in train mode
        data = {
            "x_rel": torch.tensor([[1.3, 2.7, 0.1], [0.5, 1.2, 2.8]], requires_grad=True),
            "p": torch.randn(2, 4),
        }
        result = layer(data)
        result["x"].sum().backward()
        for param in layer.net.parameters():
            if param.requires_grad:
                assert param.grad is not None

    def test_multi_var(self, multi_vars):
        net = _make_net(4, 5)
        layer = StochasticDynamicThresholdRounding(
            multi_vars, param_keys=["p"], net=net
        )
        layer.eval()
        data = {
            "x_rel": torch.tensor([[1.3, 2.7, 0.1]]),
            "y_rel": torch.tensor([[0.8, 0.2]]),
            "p": torch.randn(1, 4),
        }
        result = layer(data)
        assert result["x"].shape == (1, 3)
        assert result["y"].shape == (1, 2)


# ── TestAdaptiveSelectionRounding ────────────────────────────────────

class TestAdaptiveSelectionRounding:
    """Test AdaptiveSelectionRounding layer."""

    def test_output_shape(self, int_var):
        net = _make_net(4, 3)
        layer = AdaptiveSelectionRounding(
            int_var, param_keys=["p"], net=net
        )
        data = {"x_rel": torch.randn(8, 3), "p": torch.randn(8, 4)}
        result = layer(data)
        assert result["x"].shape == (8, 3)

    def test_integer_rounding(self, int_var):
        """Integer variables should be rounded."""
        net = _make_net(4, 3)
        layer = AdaptiveSelectionRounding(
            int_var, param_keys=["p"], net=net
        )
        layer.eval()
        data = {"x_rel": torch.tensor([[1.3, 2.7, 0.1]]), "p": torch.randn(1, 4)}
        result = layer(data)
        x = result["x"]
        assert torch.allclose(x, x.round(), atol=0.01)

    def test_binary_rounding(self, bin_var):
        """Binary variables should be 0 or 1."""
        net = _make_net(4, 3)
        layer = AdaptiveSelectionRounding(
            bin_var, param_keys=["p"], net=net
        )
        layer.eval()
        data = {"x_rel": torch.tensor([[0.3, 0.7, 0.5]]), "p": torch.randn(1, 4)}
        result = layer(data)
        x = result["x"]
        assert torch.all((x == 0) | (x == 1))

    def test_gradient_flow_through_net(self, int_var):
        """Gradients should flow through network parameters."""
        net = _make_net(4, 3)
        layer = AdaptiveSelectionRounding(
            int_var, param_keys=["p"], net=net
        )
        layer.train()
        # Batch size >= 2 required for BatchNorm in train mode
        data = {
            "x_rel": torch.tensor([[1.3, 2.7, 0.1], [0.5, 1.2, 2.8]], requires_grad=True),
            "p": torch.randn(2, 4),
        }
        result = layer(data)
        result["x"].sum().backward()
        for param in layer.net.parameters():
            if param.requires_grad:
                assert param.grad is not None

    def test_has_learnable_params(self, int_var):
        """Should have learnable network parameters."""
        net = _make_net(4, 3)
        layer = AdaptiveSelectionRounding(
            int_var, param_keys=["p"], net=net
        )
        params = list(layer.parameters())
        assert len(params) > 0

    def test_multi_var(self, multi_vars):
        """Multi-variable with shared network."""
        net = _make_net(4, 5)
        layer = AdaptiveSelectionRounding(
            multi_vars, param_keys=["p"], net=net
        )
        layer.eval()
        data = {
            "x_rel": torch.tensor([[1.3, 2.7, 0.1]]),
            "y_rel": torch.tensor([[0.8, 0.2]]),
            "p": torch.randn(1, 4),
        }
        result = layer(data)
        assert result["x"].shape == (1, 3)
        assert result["y"].shape == (1, 2)
        assert torch.all((result["y"] == 0) | (result["y"] == 1))

    def test_continuous_update(self, mixed_var):
        """With continuous_update, continuous variables should be adjusted."""
        net = _make_net(4, 5)
        layer = AdaptiveSelectionRounding(
            mixed_var, param_keys=["p"], net=net,
            continuous_update=True,
        )
        layer.eval()
        data = {
            "x_rel": torch.tensor([[0.5, 1.7, 0.8, 3.2, 0.3]]),
            "p": torch.randn(1, 4),
        }
        result = layer(data)
        assert result["x"].shape == (1, 5)

    def test_batched(self, int_var):
        net = _make_net(4, 3)
        layer = AdaptiveSelectionRounding(
            int_var, param_keys=["p"], net=net
        )
        data = {"x_rel": torch.randn(16, 3), "p": torch.randn(16, 4)}
        result = layer(data)
        assert result["x"].shape == (16, 3)


# ── TestStochasticAdaptiveSelectionRounding ──────────────────────────

class TestStochasticAdaptiveSelectionRounding:
    """Test StochasticAdaptiveSelectionRounding layer."""

    def test_inherits_adaptive_selection(self):
        """Should be a subclass of AdaptiveSelectionRounding."""
        assert issubclass(
            StochasticAdaptiveSelectionRounding, AdaptiveSelectionRounding
        )

    def test_output_shape(self, int_var):
        net = _make_net(4, 3)
        layer = StochasticAdaptiveSelectionRounding(
            int_var, param_keys=["p"], net=net
        )
        data = {"x_rel": torch.randn(8, 3), "p": torch.randn(8, 4)}
        result = layer(data)
        assert result["x"].shape == (8, 3)

    def test_binary_rounding(self, bin_var):
        """Binary variables should be 0 or 1."""
        net = _make_net(4, 3)
        layer = StochasticAdaptiveSelectionRounding(
            bin_var, param_keys=["p"], net=net
        )
        layer.eval()
        data = {"x_rel": torch.tensor([[0.3, 0.7, 0.5]]), "p": torch.randn(1, 4)}
        result = layer(data)
        x = result["x"]
        assert torch.all((x == 0) | (x == 1))

    def test_train_stochastic(self, int_var):
        """Train mode should produce different results across calls."""
        net = _make_net(4, 3)
        layer = StochasticAdaptiveSelectionRounding(
            int_var, param_keys=["p"], net=net
        )
        layer.train()
        # Batch size >= 2 required for BatchNorm in train mode
        p = torch.randn(2, 4)
        results = []
        for _ in range(10):
            data = {"x_rel": torch.tensor([[1.5, 2.5, 0.5], [0.3, 1.8, 2.1]]), "p": p}
            results.append(layer(data)["x"])
        assert not all(torch.allclose(results[0], r) for r in results[1:])

    def test_gradient_flow(self, int_var):
        """Gradients should flow through Gumbel-Softmax path."""
        net = _make_net(4, 3)
        layer = StochasticAdaptiveSelectionRounding(
            int_var, param_keys=["p"], net=net
        )
        layer.train()
        # Batch size >= 2 required for BatchNorm in train mode
        data = {
            "x_rel": torch.tensor([[1.3, 2.7, 0.1], [0.5, 1.2, 2.8]], requires_grad=True),
            "p": torch.randn(2, 4),
        }
        result = layer(data)
        result["x"].sum().backward()
        for param in layer.net.parameters():
            if param.requires_grad:
                assert param.grad is not None

    def test_multi_var(self, multi_vars):
        net = _make_net(4, 5)
        layer = StochasticAdaptiveSelectionRounding(
            multi_vars, param_keys=["p"], net=net
        )
        layer.eval()
        data = {
            "x_rel": torch.tensor([[1.3, 2.7, 0.1]]),
            "y_rel": torch.tensor([[0.8, 0.2]]),
            "p": torch.randn(1, 4),
        }
        result = layer(data)
        assert result["x"].shape == (1, 3)
        assert result["y"].shape == (1, 2)


# ── TestRoundingNodeExport ───────────────────────────────────────────

class TestRoundingNodeExport:
    """Test that rounding nodes are exported from neuround.rounding."""

    def test_import_from_rounding(self):
        from neuround.rounding import (
            RoundingNode,
            STERounding,
            DynamicThresholdRounding,
            StochasticDynamicThresholdRounding,
            AdaptiveSelectionRounding,
            StochasticAdaptiveSelectionRounding,
        )
        assert RoundingNode is not None
        assert STERounding is not None
        assert DynamicThresholdRounding is not None
        assert StochasticDynamicThresholdRounding is not None
        assert AdaptiveSelectionRounding is not None
        assert StochasticAdaptiveSelectionRounding is not None

    def test_all_are_nodes(self):
        """All rounding nodes should be Node subclasses."""
        assert issubclass(STERounding, Node)
        assert issubclass(DynamicThresholdRounding, Node)
        assert issubclass(StochasticDynamicThresholdRounding, Node)
        assert issubclass(AdaptiveSelectionRounding, Node)
        assert issubclass(StochasticAdaptiveSelectionRounding, Node)

    def test_base_is_abstract(self):
        """RoundingNode should not be directly instantiable."""
        var = _make_var("x", 3, integer_indices=[0, 1, 2])
        with pytest.raises(TypeError):
            RoundingNode(var)
