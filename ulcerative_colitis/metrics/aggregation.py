import torch
import torch.nn.functional as F
from torch import Tensor


def max_aggregation(preds: list[Tensor]) -> Tensor:
    return F.one_hot(torch.stack(preds).argmax(dim=1).max(), 5).float().unsqueeze(0)


def mean_aggregation(preds: list[Tensor]) -> Tensor:
    return torch.stack(preds).mean(dim=0).unsqueeze(0)


def targets_aggregation(targets: list[Tensor]) -> Tensor:
    return targets[0].unsqueeze(0)
