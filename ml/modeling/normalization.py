import torch
from torch import Tensor


def sigmoid_normalization(x: Tensor) -> Tensor:
    return torch.softmax(x.sigmoid(), dim=1)
