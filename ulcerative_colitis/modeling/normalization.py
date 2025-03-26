import torch
from torch import Tensor


def sigmoid_normalization(x: Tensor) -> Tensor:
    return torch.softmax(x.sigmoid(), dim=0)


def power_normalization(x: Tensor, power: float) -> Tensor:
    x -= x.min()
    x = torch.pow(x, power)
    return x / (x.max() + 1e-6)
