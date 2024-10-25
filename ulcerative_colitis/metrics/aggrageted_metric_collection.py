from collections import defaultdict
from collections.abc import Sequence
from typing import Any, Literal

import torch
from torch import Tensor
from torch.nn import functional as F
from torchmetrics import Metric, MetricCollection


class AggregatedMetricCollection(MetricCollection):
    def __init__(
        self,
        metrics: Metric | Sequence[Metric] | dict[str, Metric],
        agg: Literal["max", "mean"],
        prefix: str = "",
    ) -> None:
        super().__init__(metrics)

        self.aggregation = agg
        self.prefix = prefix

        self.outputs: dict[str, list[Tensor]] = defaultdict(list)
        self.target: dict[str, Tensor] = {}

    def update(self, preds: Tensor, targets: Tensor, key: str) -> None:  # pylint: disable=arguments-differ
        for output, target, k in zip(preds, targets, key, strict=False):
            self.outputs[k].append(output)
            self.target[k] = target

    def compute(self) -> dict[str, Any]:
        out = {}
        for metric_name, metric in self.items():
            for outputs, targets in zip(
                self.outputs.values(), self.target.values(), strict=True
            ):
                if self.aggregation == "max":
                    metric.update(
                        F.one_hot(torch.stack(outputs).argmax(dim=1).max(), 5)
                        .float()
                        .unsqueeze(0),
                        targets.unsqueeze(0),
                    )
                elif self.aggregation == "mean":
                    metric.update(
                        torch.stack(outputs).mean(dim=0).unsqueeze(0),
                        targets.unsqueeze(0),
                    )
                else:
                    raise ValueError("Aggregation must be either 'max' or 'mean'")

            value: Tensor = metric.compute()
            out[self.prefix + metric_name] = value.item()
        print(out)
        return out
