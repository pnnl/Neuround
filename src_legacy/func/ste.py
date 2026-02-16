"""
Straight-through estimators for nondifferentiable operators
"""

import torch
from torch import nn
from torch.autograd import Function

class thresholdBinarize(nn.Module):
    """
    An autograd function smoothly rounds the elements in `x` based on the
    corresponding values in `threshold` using a sigmoid function.
    """
    def __init__(self, slope=10):
        super(thresholdBinarize, self).__init__()
        self.slope = slope

    def forward(self, x, threshold):
        # ensure the threshold_tensor values are between 0 and 1
        threshold = torch.clamp(threshold, 0, 1)
        # hard rounding
        hard_round = (x >= threshold).float()
        # calculate the difference and apply the sigmoid function
        diff = x - threshold
        smoothed_round = torch.sigmoid(self.slope * diff)
        # apply the STE trick to keep the gradient
        return hard_round + (smoothed_round - smoothed_round.detach())


class diffGumbelBinarize(nn.Module):
    """
    An autograd function to binarize numbers using the Gumbel-Softmax trick,
    allowing gradients to be backpropagated through discrete variables.
    """
    def __init__(self, temperature=1.0, eps=1e-9):
        super(diffGumbelBinarize, self).__init__()
        self.temperature = temperature
        self.eps = eps

    def forward(self, x):
        # train mode
        if self.training:
            # Gumbel sampling
            gumbel_noise0 = self._gumbelSample(x)
            gumbel_noise1 = self._gumbelSample(x)
            # sigmoid with Gumbel
            noisy_diff = x + gumbel_noise1 - gumbel_noise0
            soft_sample = torch.sigmoid(noisy_diff / self.temperature)
            # hard rounding
            hard_sample = (soft_sample > 0.5).float()
            # apply the STE trick to keep the gradient
            return hard_sample + (soft_sample - soft_sample.detach())
        # eval mode
        else:
            # use a temperature-scaled sigmoid in evaluation mode for consistency
            return (torch.sigmoid(x / self.temperature) > 0.5).float()

    def _gumbelSample(self, x):
        """
        Generates Gumbel noise based on the input shape and device
        """
        u = torch.rand_like(x)
        return - torch.log(- torch.log(u + self.eps) + self.eps)


class diffBinarize(nn.Module):
    """
    An autograd model to binarize numbers
    """
    def __init__(self):
        super(diffBinarize, self).__init__()

    def forward(self, x):
        x = _binarizeFuncSTE.apply(x)
        return x


class _binarizeFuncSTE(Function):
    """
    An autograd binarize function with straight-through estimator
    """
    @staticmethod
    def forward(ctx, input):
        # clipped ReLU
        logit = torch.clamp(input, min=-1, max=1)
        # save for backward pass
        ctx.save_for_backward(input)
        # binarize to 0 or 1
        return (logit >= 0).float()

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        # copy
        grad_input = grad_output.clone()
        # modify the gradients
        grad_input[torch.abs(input) < 1] = 0
        return grad_input


class diffFloor(nn.Module):
    """
    An autograd model to floor numbers that applies a straight-through estimator
    for the backward pass.
    """
    def __init__(self):
        super(diffFloor, self).__init__()

    def forward(self, x):
        # floor
        x_floor = torch.floor(x).float()
        # apply the STE trick to keep the gradient
        return x_floor + (x - x.detach())


if __name__ == "__main__":

    # add system path
    import sys
    import os
    sys.path.append(os.path.abspath("."))
    sys.path.append(os.path.abspath(".."))

    x = torch.tensor([0.2, 0.5, 0.7])
    v = torch.tensor([0.1, 0.5, 0.8], requires_grad=True)
    round_func = thresholdBinarize()
    print("Input tensor:", x)
    print("Threshold:", v)
    res = round_func(x, v)
    print(res)
    res.backward(torch.ones_like(res))
    print("Grad:", v.grad)
    print()

    # input tensor
    x = torch.tensor([[-1.2, -0.5, 0.0, 0.5, 1.2]], requires_grad=True)
    print("Input tensor:", x)
    print()

    # floor
    floor = diffFloor()
    x_flr = floor(x)
    print("Floor vector", x_flr)
    loss = x_flr.sum()
    loss.backward()
    print("Grad", x.grad)

    # Reset gradients
    x.grad.zero_()
    print()

    # binarize
    bin = diffBinarize()
    x_bin = bin(x)
    print("Binarize vector", x_bin)
    loss = x_bin.sum()
    loss.backward()
    print("Grad:", x.grad)

    # Reset gradients
    x.grad.zero_()
    print()

    # binarize
    bin = diffGumbelBinarize()
    x_bin = bin(x)
    print("Gumbel Binarize vector", x_bin)
    loss = x_bin.sum()
    loss.backward()
    print("Grad:", x.grad)
