"""
Network blocks for NeuroPMINLP.
"""

import torch.nn as nn


class MLPBnDrop(nn.Module):
    """
    MLP with BatchNorm and Dropout.
    Follows neuromancer MLP interface (insize, outsize, hsizes).

    Architecture:
        [Linear -> nonlin -> BatchNorm -> Dropout] * N -> Linear

    Args:
        insize: Input dimension
        outsize: Output dimension
        hsizes: List of hidden layer sizes
        nonlin: Activation function class (default: nn.ReLU)
        dropout: Dropout probability (default: 0.2, 0 to disable)
        bnorm: Enable BatchNorm (default: True)
        bias: Use bias in linear layers (default: True)

    Example:
        >>> net = MLPBnDrop(insize=20, outsize=10, hsizes=[64]*4)
        >>> net = MLPBnDrop(insize=20, outsize=10, hsizes=[64]*4,
        ...                 dropout=0.0, bnorm=False)  # equivalent to MLP
    """

    def __init__(self, insize, outsize, hsizes,
                 nonlin=nn.ReLU, dropout=0.2, bnorm=True, bias=True):
        super().__init__()
        sizes = [insize] + hsizes + [outsize]
        layers = []
        for i in range(len(sizes) - 2):
            layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=bias))
            layers.append(nonlin())
            if bnorm:
                layers.append(nn.BatchNorm1d(sizes[i + 1]))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
        # last layer: Linear only
        layers.append(nn.Linear(sizes[-2], sizes[-1], bias=bias))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
