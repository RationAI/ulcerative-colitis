from collections import defaultdict
from collections.abc import Callable, Sequence
from typing import Any

from torch import Tensor
from torchmetrics import Metric, MetricCollection


class AggregatedMetricCollection(MetricCollection):
    def __init__(
        self,
        metrics: Metric | Sequence[Metric] | dict[str, Metric],
        aggregation_preds: Callable[[list[Tensor]], Tensor],
        aggregation_targets: Callable[[list[Tensor]], Tensor],
        prefix: str = "",
    ) -> None:
        super().__init__(metrics)

        self.aggregation_preds = aggregation_preds
        self.aggregation_targets = aggregation_targets
        self.prefix = prefix

        self.preds: dict[str, list[Tensor]] = defaultdict(list)
        self.targets: dict[str, list[Tensor]] = defaultdict(list)

    def update(self, preds: Tensor, targets: Tensor, key: str) -> None:  # pylint: disable=arguments-differ
        for pred, target, k in zip(preds, targets, key, strict=False):
            self.preds[k].append(pred)
            self.targets[k].append(target)

    def compute(self) -> dict[str, Any]:
        out = {}
        for metric_name, metric in self.items():
            for output, target in zip(
                self.preds.values(), self.targets.values(), strict=True
            ):
                metric.update(
                    self.aggregation_preds(output), self.aggregation_targets(target)
                )

            value: Tensor = metric.compute()
            out[self.prefix + metric_name] = value.item()
        return out
