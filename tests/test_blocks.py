"""
Unit tests for MLPBnDrop network block.
"""

import pytest
import torch
import torch.nn as nn

from neuropminlp.blocks import MLPBnDrop


class TestMLPBnDrop:
    """Test MLPBnDrop with various configurations."""

    def test_forward_shape(self):
        net = MLPBnDrop(insize=20, outsize=10, hsizes=[64, 64])
        x = torch.randn(8, 20)
        y = net(x)
        assert y.shape == (8, 10)

    def test_default_params(self):
        net = MLPBnDrop(insize=10, outsize=5, hsizes=[32, 32])
        layer_types = [type(m) for m in net.net]
        assert nn.BatchNorm1d in layer_types
        assert nn.Dropout in layer_types
        assert nn.ReLU in layer_types

    def test_no_bnorm(self):
        net = MLPBnDrop(insize=10, outsize=5, hsizes=[32], bnorm=False)
        layer_types = [type(m) for m in net.net]
        assert nn.BatchNorm1d not in layer_types

    def test_no_dropout(self):
        net = MLPBnDrop(insize=10, outsize=5, hsizes=[32], dropout=0.0)
        layer_types = [type(m) for m in net.net]
        assert nn.Dropout not in layer_types

    def test_no_bnorm_no_dropout(self):
        net = MLPBnDrop(insize=10, outsize=5, hsizes=[32],
                        bnorm=False, dropout=0.0)
        layer_types = [type(m) for m in net.net]
        assert nn.BatchNorm1d not in layer_types
        assert nn.Dropout not in layer_types
        # Should only have Linear and ReLU layers
        for m in net.net:
            assert isinstance(m, (nn.Linear, nn.ReLU))

    def test_custom_nonlin(self):
        net = MLPBnDrop(insize=10, outsize=5, hsizes=[32], nonlin=nn.Tanh)
        layer_types = [type(m) for m in net.net]
        assert nn.Tanh in layer_types
        assert nn.ReLU not in layer_types

    def test_bias_false(self):
        net = MLPBnDrop(insize=10, outsize=5, hsizes=[32], bias=False)
        for m in net.net:
            if isinstance(m, nn.Linear):
                assert m.bias is None

    def test_single_hidden(self):
        net = MLPBnDrop(insize=10, outsize=5, hsizes=[32])
        linears = [m for m in net.net if isinstance(m, nn.Linear)]
        assert len(linears) == 2  # hidden + output
        assert linears[0].in_features == 10
        assert linears[0].out_features == 32
        assert linears[1].in_features == 32
        assert linears[1].out_features == 5

    def test_multiple_hidden(self):
        net = MLPBnDrop(insize=10, outsize=5, hsizes=[64, 32, 16])
        linears = [m for m in net.net if isinstance(m, nn.Linear)]
        assert len(linears) == 4  # 3 hidden + 1 output
        assert linears[0].in_features == 10
        assert linears[0].out_features == 64
        assert linears[1].in_features == 64
        assert linears[1].out_features == 32
        assert linears[2].in_features == 32
        assert linears[2].out_features == 16
        assert linears[3].in_features == 16
        assert linears[3].out_features == 5

    def test_train_eval_mode(self):
        net = MLPBnDrop(insize=10, outsize=5, hsizes=[32])
        x = torch.randn(8, 10)
        # Train mode: dropout active, outputs vary
        net.train()
        out_train_1 = net(x)
        out_train_2 = net(x)
        # Eval mode: dropout disabled, outputs deterministic
        net.eval()
        out_eval_1 = net(x)
        out_eval_2 = net(x)
        assert torch.equal(out_eval_1, out_eval_2)

    def test_export(self):
        import neuropminlp
        assert hasattr(neuropminlp, "MLPBnDrop")
        assert "MLPBnDrop" in neuropminlp.__all__
        assert neuropminlp.MLPBnDrop is MLPBnDrop
