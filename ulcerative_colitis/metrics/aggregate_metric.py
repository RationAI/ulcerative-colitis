from collections import defaultdict
from collections.abc import Sequence
from typing import Any, Literal

import torch
from torch import Tensor
from torch.nn import functional as F
from torchmetrics import Metric, MetricCollection


class MetricAggregator(MetricCollection):
    def __init__(
        self,
        metrics: Metric | Sequence[Metric] | dict[str, Metric],
        agg: Literal["max", "mean"],
        prefix: str = "",
        class_names: list[str] | None = None,
    ) -> None:
        super().__init__(metrics)

        self.aggregation = agg
        self.prefix = prefix
        self.class_names = class_names

        self.outputs: dict[str, list[Tensor]] = defaultdict(list)
        self.target: dict[str, Tensor] = {}

    def update(self, outputs: Tensor, targets: Tensor, key: str) -> None:  # pylint: disable=arguments-differ
        self.outputs[key].append(outputs)
        self.target[key] = targets

    def compute(self) -> dict[str, Any]:
        out = {}
        for metric_name, metric in self.items():
            for outputs, targets in zip(
                self.outputs.values(), self.target.values(), strict=True
            ):
                if self.aggregation == "max":
                    metric.update(
                        F.one_hot(torch.stack(outputs).argmax(dim=1).max(), 5), targets
                    )
                elif self.aggregation == "mean":
                    metric.update(torch.stack(outputs).mean(dim=0), targets)
                else:
                    raise ValueError("Aggregation must be either 'max' or 'mean'")

            value: Tensor = metric.compute()
            if not value.shape:
                out[self.prefix + metric_name] = value.item()
            else:
                assert len(value.shape) == 1
                if self.class_names is None:
                    for i, v in enumerate(value):
                        out[f"{self.prefix}{metric_name}_{i}"] = v.item()
                else:
                    if len(value) != len(self.class_names):
                        raise ValueError(
                            f"Expected {len(self.class_names)} classes, got {len(value)}"
                        )
                    for class_name, v in zip(self.class_names, value, strict=True):
                        out[f"{self.prefix}{metric_name}_{class_name}"] = v.item()
        return out
