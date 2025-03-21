from collections.abc import Mapping
from copy import deepcopy
from typing import cast

import torch
from lightning import LightningModule
from rationai.mlkit.metrics import (
    AggregatedMetricCollection,
    MaxAggregator,
    MeanAggregator,
    MeanPoolMaxAggregator,
)
from torch import Tensor
from torch.nn import CrossEntropyLoss, ModuleDict
from torch.optim.adam import Adam
from torch.optim.optimizer import Optimizer
from torchmetrics import Metric, MetricCollection
from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassAUROC,
    MulticlassPrecision,
    MulticlassRecall,
)

from ulcerative_colitis.modeling.linear_probe import LinearProbe
from ulcerative_colitis.typing import (
    MetadataBatch,
    Output,
    PredictInput,
    TestInput,
    TrainInput,
)


class UlcerativeColitisModelLinearProbe(LightningModule):
    def __init__(self, lr: float | None = None) -> None:
        super().__init__()
        self.decode_head = LinearProbe()
        self.criterion = CrossEntropyLoss()
        self.lr = lr

        metrics: dict[str, Metric] = {
            "AUC": MulticlassAUROC(5, average="none"),
            "accuracy": MulticlassAccuracy(5),
            "precision": MulticlassPrecision(5, average="none"),
            "recall": MulticlassRecall(5, average="none"),
        }

        self.val_metrics: dict[str, MetricCollection] = cast(
            "dict",
            ModuleDict(
                {
                    "tiles_all": MetricCollection(
                        deepcopy(metrics), prefix="validation/tiles/"
                    ),
                }
            ),
        )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        max_aggregator = MaxAggregator().to(device)
        mean_aggregator = MeanAggregator().to(device)
        mean_pool_max_aggregator = MeanPoolMaxAggregator(2, 512, 256).to(device)
        self.test_metrics: dict[str, MetricCollection] = cast(
            "dict",
            ModuleDict(
                {
                    "tiles_all": MetricCollection(
                        deepcopy(metrics), prefix="test/tiles/"
                    ),
                    "slides_max1": AggregatedMetricCollection(
                        deepcopy(metrics),
                        max_aggregator,
                        prefix="test/slides/max1/",
                    ),
                    "slides_max2": AggregatedMetricCollection(
                        deepcopy(metrics),
                        mean_pool_max_aggregator,
                        prefix="test/slides/max2/",
                    ),
                    "slides_mean": AggregatedMetricCollection(
                        deepcopy(metrics),
                        mean_aggregator,
                        prefix="test/slides/mean/",
                    ),
                }
            ),
        )

    def forward(self, x: Tensor) -> Output:  # pylint: disable=arguments-differ
        return self.decode_head(x)

    def training_step(self, batch: TrainInput) -> Tensor:  # pylint: disable=arguments-differ
        inputs, targets, _ = batch
        inputs_shape = inputs.shape  # (batch_size, inner_batch_size, embedding_size)
        inputs = inputs.view(inputs_shape[0] * inputs_shape[1], *inputs_shape[2:])
        outputs = cast("torch.Tensor", self(inputs))
        outputs = outputs.view(inputs_shape[0], inputs_shape[1], *outputs.shape[1:])

        loss = self.criterion(outputs.mean(dim=1), targets)
        self.log("train/loss", loss, on_step=True, prog_bar=True)

        return loss

    def validation_step(self, batch: TestInput) -> None:  # pylint: disable=arguments-differ
        inputs, targets, metadata = batch
        inputs_shape = inputs.shape  # (batch_size, inner_batch_size, embedding_size)
        inputs = inputs.view(inputs_shape[0] * inputs_shape[1], *inputs_shape[2:])
        outputs = cast("torch.Tensor", self(inputs))
        outputs = outputs.view(inputs_shape[0], inputs_shape[1], *outputs.shape[1:])
        outputs = outputs.mean(dim=1)

        loss = self.criterion(outputs, targets)
        self.log("validation/loss", loss, on_epoch=True, prog_bar=True)

        # metadata not correct, but not used in this function
        # self.update_metrics(self.val_metrics, outputs, targets, metadata)
        self.val_metrics["tiles_all"].update(outputs, targets)
        self.log_metrics(self.val_metrics)

    def test_step(self, batch: TestInput) -> None:  # pylint: disable=arguments-differ
        inputs, targets, metadata = batch
        outputs = self(inputs)

        self.update_metrics(self.test_metrics, outputs, targets, metadata)
        self.log_metrics(self.test_metrics)

    def predict_step(  # pylint: disable=arguments-differ
        self, batch: PredictInput, batch_idx: int, dataloader_idx: int = 0
    ) -> Output:
        inputs, _ = batch
        outputs = self(inputs)

        return outputs

    def configure_optimizers(self) -> Optimizer:
        if self.lr is None:
            raise ValueError("Learning rate must be set for training.")
        return Adam(self.parameters(), lr=self.lr)

    def log_dict(self, dictionary: MetricCollection, *args, **kwargs) -> None:
        for name, result in dictionary.compute().items():
            result = cast("Tensor", result)
            if result.shape:
                for i, value in enumerate(result):
                    self.log(f"{name}/{i}", value, *args, **kwargs)
            else:
                self.log(name, result, *args, **kwargs)

    def update_metrics(
        self,
        metrics: Mapping[str, MetricCollection],
        outputs: Tensor,
        targets: Tensor,
        metadata: MetadataBatch,
    ) -> None:
        for name, metric in metrics.items():
            if "slide" in name:
                metric.update(
                    outputs if "max2" in name else outputs.squeeze(-1),
                    targets.squeeze(-1),
                    keys=metadata["slide"],
                    x=metadata["x"],
                    y=metadata["y"],
                )
            else:
                metric.update(outputs, targets)

    def log_metrics(self, metrics: dict[str, MetricCollection]) -> None:
        for metric in metrics.values():
            self.log_dict(metric, on_epoch=True)
