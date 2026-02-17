"""
Unit tests for VarType enum and variable() function.
"""

import pytest
import torch
from neuromancer.constraint import Variable

from neuropminlp.variable import VarType, variable


class TestVarType:
    """Test VarType enumeration."""

    def test_values(self):
        assert VarType.CONTINUOUS.value == "continuous"
        assert VarType.INTEGER.value == "integer"
        assert VarType.BINARY.value == "binary"

    def test_repr(self):
        assert repr(VarType.CONTINUOUS) == "VarType.CONTINUOUS"
        assert repr(VarType.INTEGER) == "VarType.INTEGER"
        assert repr(VarType.BINARY) == "VarType.BINARY"

    def test_all_members(self):
        members = list(VarType)
        assert len(members) == 3


class TestVariableSymbolic:
    """Test variable() without type info (symbolic variable)."""

    def test_plain_variable(self):
        x = variable("x")
        assert x.key == "x"

    def test_no_type_metadata(self):
        x = variable("x")
        assert not hasattr(x, "var_types")
        assert not hasattr(x, "num_vars")
        assert not hasattr(x, "relaxed")


class TestVariableIndexBased:
    """Test variable() with index-based type specification."""

    def test_all_continuous(self):
        x = variable("x", num_vars=5)
        assert x.num_vars == 5
        assert x.var_types == [VarType.CONTINUOUS] * 5
        assert x.integer_indices == []
        assert x.binary_indices == []
        assert x.continuous_indices == [0, 1, 2, 3, 4]

    def test_integer_indices(self):
        x = variable("x", num_vars=5, integer_indices=[0, 2, 4])
        assert x.integer_indices == [0, 2, 4]
        assert x.continuous_indices == [1, 3]
        assert x.binary_indices == []
        assert x.var_types[0] == VarType.INTEGER
        assert x.var_types[1] == VarType.CONTINUOUS
        assert x.var_types[2] == VarType.INTEGER

    def test_binary_indices(self):
        x = variable("x", num_vars=4, binary_indices=[1, 3])
        assert x.binary_indices == [1, 3]
        assert x.continuous_indices == [0, 2]
        assert x.integer_indices == []

    def test_mixed_indices(self):
        x = variable("x", num_vars=6,
                      integer_indices=[0, 1],
                      binary_indices=[4, 5])
        assert x.integer_indices == [0, 1]
        assert x.binary_indices == [4, 5]
        assert x.continuous_indices == [2, 3]
        assert x.num_vars == 6

    def test_all_integer(self):
        x = variable("x", num_vars=3, integer_indices=[0, 1, 2])
        assert x.integer_indices == [0, 1, 2]
        assert x.continuous_indices == []
        assert x.binary_indices == []


class TestVariableExplicitTypes:
    """Test variable() with explicit var_types list."""

    def test_explicit_types(self):
        types = [VarType.INTEGER, VarType.CONTINUOUS, VarType.BINARY]
        x = variable("x", var_types=types)
        assert x.var_types == types
        assert x.num_vars == 3
        assert x.integer_indices == [0]
        assert x.continuous_indices == [1]
        assert x.binary_indices == [2]

    def test_all_continuous_explicit(self):
        types = [VarType.CONTINUOUS] * 4
        x = variable("x", var_types=types)
        assert x.continuous_indices == [0, 1, 2, 3]
        assert x.integer_indices == []
        assert x.binary_indices == []


class TestVariableRelaxed:
    """Test auto-creation of relaxed variable."""

    def test_relaxed_with_discrete(self):
        x = variable("x", num_vars=3, integer_indices=[0])
        assert x.relaxed.key == "x_rel"
        assert x.relaxed is not x

    def test_relaxed_with_binary(self):
        x = variable("x", num_vars=3, binary_indices=[0])
        assert x.relaxed.key == "x_rel"

    def test_relaxed_pure_continuous(self):
        x = variable("x", num_vars=3)
        assert x.relaxed is x

    def test_relaxed_key_preserved(self):
        x = variable("x", num_vars=5, integer_indices=[0, 1, 2])
        assert x.key == "x"
        assert x.relaxed.key == "x_rel"


class TestVariableErrors:
    """Test error handling in variable()."""

    def test_indices_without_num_vars(self):
        with pytest.raises(ValueError, match="num_vars is required"):
            variable("x", integer_indices=[0])

    def test_binary_indices_without_num_vars(self):
        with pytest.raises(ValueError, match="num_vars is required"):
            variable("x", binary_indices=[0])

    def test_var_types_with_num_vars_conflict(self):
        with pytest.raises(ValueError, match="Cannot specify both"):
            variable("x", num_vars=3,
                     var_types=[VarType.CONTINUOUS] * 3)

    def test_var_types_with_indices_conflict(self):
        with pytest.raises(ValueError, match="Cannot specify both"):
            variable("x", var_types=[VarType.CONTINUOUS],
                     integer_indices=[0])

    def test_integer_index_out_of_range(self):
        with pytest.raises(ValueError, match="out of range"):
            variable("x", num_vars=3, integer_indices=[5])

    def test_binary_index_out_of_range(self):
        with pytest.raises(ValueError, match="out of range"):
            variable("x", num_vars=3, binary_indices=[3])

    def test_negative_index(self):
        with pytest.raises(ValueError, match="out of range"):
            variable("x", num_vars=3, integer_indices=[-1])

    def test_overlap_integer_binary(self):
        with pytest.raises(ValueError, match="appear in both"):
            variable("x", num_vars=5,
                     integer_indices=[0, 1],
                     binary_indices=[1, 2])


class TestVariableIsNeuromancer:
    """Test that variable() returns a proper neuromancer Variable."""

    def test_instance_type(self):
        x = variable("x")
        assert isinstance(x, Variable)

    def test_typed_variable_instance_type(self):
        x = variable("x", num_vars=3, integer_indices=[0])
        assert isinstance(x, Variable)
        assert isinstance(x.relaxed, Variable)


class TestVariableComputationGraph:
    """Test that variables work in neuromancer computation graphs."""

    def test_arithmetic_builds_graph(self):
        x = variable("x")
        y = variable("y")
        z = x + y
        assert isinstance(z, Variable)

    def test_forward_with_datadict(self):
        x = variable("x")
        y = variable("y")
        z = x + y
        data = {"x": torch.tensor([1.0, 2.0]), "y": torch.tensor([3.0, 4.0])}
        result = z(data)
        assert torch.allclose(result, torch.tensor([4.0, 6.0]))

    def test_matmul(self):
        x = variable("x")
        W = torch.randn(3, 2)
        z = x @ W
        data = {"x": torch.ones(1, 3)}
        result = z(data)
        assert result.shape == (1, 2)

    def test_typed_variable_in_graph(self):
        """Typed variables should still work as neuromancer variables."""
        x = variable("x", num_vars=3, integer_indices=[0, 1, 2])
        y = variable("y")
        z = x + y
        data = {"x": torch.tensor([1.0, 2.0, 3.0]),
                "y": torch.tensor([10.0, 20.0, 30.0])}
        result = z(data)
        assert torch.allclose(result, torch.tensor([11.0, 22.0, 33.0]))

    def test_relaxed_variable_in_graph(self):
        """Relaxed variable should work as computation graph input."""
        x = variable("x", num_vars=2, integer_indices=[0, 1])
        # relaxed key is "x_rel", used as input to smap/RoundingLayer
        y = x.relaxed + 1.0
        data = {"x_rel": torch.tensor([0.5, 0.7])}
        result = y(data)
        assert torch.allclose(result, torch.tensor([1.5, 1.7]))

    def test_sub(self):
        x = variable("x")
        y = variable("y")
        z = x - y
        data = {"x": torch.tensor([5.0]), "y": torch.tensor([3.0])}
        assert torch.allclose(z(data), torch.tensor([2.0]))

    def test_mul(self):
        x = variable("x")
        z = x * 2.0
        data = {"x": torch.tensor([3.0, 4.0])}
        assert torch.allclose(z(data), torch.tensor([6.0, 8.0]))

    def test_div(self):
        x = variable("x")
        z = x / 2.0
        data = {"x": torch.tensor([6.0, 8.0])}
        assert torch.allclose(z(data), torch.tensor([3.0, 4.0]))

    def test_pow(self):
        x = variable("x")
        z = x ** 2
        data = {"x": torch.tensor([3.0])}
        assert torch.allclose(z(data), torch.tensor([9.0]))

    def test_neg(self):
        x = variable("x")
        z = -x
        data = {"x": torch.tensor([1.0, -2.0])}
        assert torch.allclose(z(data), torch.tensor([-1.0, 2.0]))

    def test_chained_ops(self):
        """Test chained operations build correct graph."""
        x = variable("x")
        y = variable("y")
        z = (x + y) * 2.0 - 1.0
        data = {"x": torch.tensor([1.0]), "y": torch.tensor([2.0])}
        assert torch.allclose(z(data), torch.tensor([5.0]))

    def test_torch_sin(self):
        x = variable("x")
        z = torch.sin(x)
        data = {"x": torch.tensor([0.0, torch.pi / 2])}
        result = z(data)
        assert torch.allclose(result, torch.tensor([0.0, 1.0]), atol=1e-6)

    def test_torch_cos(self):
        x = variable("x")
        z = torch.cos(x)
        data = {"x": torch.tensor([0.0, torch.pi])}
        result = z(data)
        assert torch.allclose(result, torch.tensor([1.0, -1.0]), atol=1e-6)

    def test_torch_exp(self):
        x = variable("x")
        z = torch.exp(x)
        data = {"x": torch.tensor([0.0, 1.0])}
        result = z(data)
        expected = torch.tensor([1.0, torch.e])
        assert torch.allclose(result, expected, atol=1e-5)

    def test_constraint_creation(self):
        """Comparison operators should create Constraint objects."""
        from neuromancer.constraint import Constraint
        x = variable("x")
        con = x <= 5.0
        assert isinstance(con, Constraint)

    def test_objective_creation(self):
        """minimize() should create an Objective."""
        from neuromancer.constraint import Objective
        x = variable("x")
        obj = (x ** 2).minimize()
        assert isinstance(obj, Objective)


class TestVariableNodeIntegration:
    """Test variable() integration with neuromancer Node."""

    def test_node_with_plain_variable(self):
        """Node should accept variable keys for input/output."""
        from neuromancer.system import Node
        import torch.nn as nn

        net = nn.Linear(3, 2)
        x = variable("x")
        node = Node(net, [x.key], ["y_hat"], name="predictor")
        data = {x.key: torch.randn(4, 3)}
        out = node(data)
        assert "y_hat" in out
        assert out["y_hat"].shape == (4, 2)

    def test_node_with_relaxed_key(self):
        """Node should use relaxed key as input, var key as output."""
        from neuromancer.system import Node
        import torch.nn as nn

        net = nn.Linear(3, 2)
        x = variable("x", num_vars=2, integer_indices=[0, 1])
        # smap pattern: network takes x_rel as input, outputs x_rel
        node = Node(net, [x.relaxed.key], [x.relaxed.key], name="smap")
        data = {"x_rel": torch.randn(4, 3)}
        out = node(data)
        assert x.relaxed.key in out
        assert out["x_rel"].shape == (4, 2)

    def test_node_chain_with_variables(self):
        """Chain multiple Nodes using variable keys."""
        from neuromancer.system import Node
        import torch.nn as nn

        x = variable("x", num_vars=3, integer_indices=[0, 1, 2])
        # Node 1: smap outputs relaxed solution
        net = nn.Linear(2, 3)
        smap = Node(net, ["b"], [x.relaxed.key], name="smap")
        # Node 2: identity as placeholder for rounding (x_rel -> x)
        rnd = Node(lambda d: d, [x.relaxed.key], [x.key], name="round")

        data = {"b": torch.randn(4, 2)}
        data = smap(data)
        assert x.relaxed.key in data  # "x_rel"
        data = rnd(data)
        assert x.key in data  # "x"


class TestVariableExport:
    """Test that VarType and variable are exported from neuropminlp."""

    def test_vartype_export(self):
        import neuropminlp
        assert hasattr(neuropminlp, "VarType")
        assert "VarType" in neuropminlp.__all__
        assert neuropminlp.VarType is VarType

    def test_variable_export(self):
        import neuropminlp
        assert hasattr(neuropminlp, "variable")
        assert "variable" in neuropminlp.__all__
        assert neuropminlp.variable is variable
