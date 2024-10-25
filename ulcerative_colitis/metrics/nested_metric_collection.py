from collections import defaultdict
from copy import deepcopy
from typing import Any

from torch import Tensor
from torchmetrics import Metric, MetricCollection


class NestedMetricCollection(MetricCollection):
    def __init__(
        self,
        metrics: Metric | MetricCollection,
        key_name: str = "key",
        class_names: list[str] | None = None,
    ) -> None:
        super().__init__([])
        if isinstance(metrics, Metric):
            metrics = MetricCollection(metrics)

        self.metrics = dict(metrics.items())
        self.key_name = key_name
        self.class_names = class_names

    def update(  # pylint: disable=arguments-differ
        self, preds: Tensor, targets: Tensor, key: list[str]
    ) -> None:
        for k in key:
            for name, metric in self.metrics.items():
                new_name = f"{k}/{name}"
                if new_name not in self:
                    self.add_metrics({new_name: deepcopy(metric).to(preds.device)})

                self[new_name].update(preds, targets)

    def compute(self) -> dict[str, Any]:
        divided_metrics = defaultdict(dict)
        for name, metric in self.items():
            key, subkey = name.split("/", maxsplit=1)

            value: Tensor = metric.compute()
            if not value.shape:
                divided_metrics[key][subkey] = value.item()
            else:
                assert len(value.shape) == 1
                if self.class_names is None:
                    for i, v in enumerate(value):
                        divided_metrics[key][f"{subkey}/{i}"] = v.item()
                else:
                    if len(value) != len(self.class_names):
                        raise ValueError(
                            f"Expected {len(self.class_names)} classes, got {len(value)}"
                        )
                    for class_name, v in zip(self.class_names, value, strict=True):
                        divided_metrics[key][f"{subkey}/{class_name}"] = v.item()

        out = defaultdict(list)
        for key, metrics in divided_metrics.items():
            out[self.key_name].append(key)
            for subkey, value in metrics.items():
                out[subkey].append(value)
        return out
