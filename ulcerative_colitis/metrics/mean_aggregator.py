from typing import Any

import torch
from rationai.mlkit.metrics.aggregators import Aggregator
from torch import Tensor


class MeanAggregator(Aggregator):
    """Aggregator to compute the mean of predictions and targets."""

    def __init__(self) -> None:
        super().__init__()
        self.add_state(
            "preds",
            default=torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0]),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "targets",
            default=torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0]),
            dist_reduce_fx="sum",
        )
        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: Tensor, targets: Tensor, **kwargs: Any) -> None:
        self.preds += preds
        self.targets += targets
        self.count += 1

    def compute(self) -> tuple[Tensor, Tensor]:
        return self.preds / self.count, self.targets / self.count
