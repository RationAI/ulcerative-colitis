from collections import defaultdict
from collections.abc import Callable, Sequence
from typing import Any

from torch import Tensor
from torchmetrics import Metric, MetricCollection


class AggregatedMetricCollection(MetricCollection):
    """AggregatedMetricCollection is a MetricCollection that aggregates the predictions and targets before computing the metrics.

    Arguments:
        metrics (Metric | Sequence[Metric] | dict[str, Metric]): Metric(s) to be computed.
        aggregation_preds (Callable[[list[Tensor]], Tensor]): Function to aggregate the predictions.
        aggregation_targets (Callable[[list[Tensor]], Tensor]): Function to aggregate the targets.
        prefix (str): Prefix to add to the metric names.

    Example:
        >>> from torchmetrics import Accuracy, Precision, Recall
        >>> metrics = {
        ...     "accuracy": Accuracy(),
        ...     "precision": Precision(),
        ...     "recall": Recall(),
        ... }
        >>> def meam_aggregation(x):
        ...     return torch.stack(preds).mean(dim=0).unsqueeze(0)
        >>> def targets_aggregation(x):
        ...     return targets[0].unsqueeze(0)
        >>> agg_metrics = AggregatedMetricCollection(
        ...     metrics,
        ...     aggregation_preds=max_aggregation,
        ...     aggregation_targets=targets_aggregation,
        ... )
        >>> preds = torch.tensor([[0.1, 0.9], [0.8, 0.2], [0.3, 0.7], [0.6, 0.4]])
        >>> targets = torch.tensor([1, 1, 0, 0])
        >>> key = ["slide1", "slide1", "slide2", "slide2"]
        >>> agg_metrics.update(preds, targets, key)
        >>> agg_metrics.compute()
        {
            "accuracy": 0.5,
            "precision": 0.5,
            "recall": 0.5,
        }
    """

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
