from collections.abc import Mapping
from copy import deepcopy
from typing import cast

import torch
from lightning import LightningModule
from rationai.mlkit.metrics import (
    AggregatedMetricCollection,
    MaxAggregator,
    MeanPoolMaxAggregator,
)
from torch import Tensor
from torch.nn import BCELoss, Module, ModuleDict
from torch.optim.adam import Adam
from torch.optim.optimizer import Optimizer
from torchmetrics import Metric, MetricCollection
from torchmetrics.classification import (
    BinaryAccuracy,
    BinaryAUROC,
    BinaryPrecision,
    BinaryRecall,
)

from ulcerative_colitis.modeling.binary_classification_head import (
    BinaryClassificationHead,
)
from ulcerative_colitis.typing import (
    MetadataBatch,
    Output,
    PredictInput,
    TestInput,
    TrainInput,
)


class UlcerativeColitisModelBinary(LightningModule):
    def __init__(self, backbone: Module, lr: float) -> None:
        super().__init__()
        self.backbone = backbone
        self.decode_head = BinaryClassificationHead()
        self.criterion = BCELoss()
        self.lr = lr

        metrics: dict[str, Metric] = {
            "AUC": BinaryAUROC(),
            "accuracy": BinaryAccuracy(),
            "precision": BinaryPrecision(),
            "recall": BinaryRecall(),
        }

        max_aggregator = MaxAggregator()
        mean_pool_max_aggregator = MeanPoolMaxAggregator(2, 512, 256)
        self.val_metrics: dict[str, MetricCollection] = cast(
            dict,
            ModuleDict(
                {
                    "tiles_all": MetricCollection(
                        deepcopy(metrics), prefix="validation/tiles/"
                    ),
                    "slides_max1": AggregatedMetricCollection(
                        deepcopy(metrics),
                        max_aggregator,
                        prefix="validation/slides/max1/",
                    ),
                    "slides_max2": AggregatedMetricCollection(
                        deepcopy(metrics),
                        mean_pool_max_aggregator,
                        prefix="validation/slides/max2/",
                    ),
                }
            ),
        )

        self.test_metrics: dict[str, MetricCollection] = cast(
            dict,
            ModuleDict(
                {
                    name: metric.clone(
                        prefix=cast(str, metric.prefix).replace("validation", "test")
                    )
                    for name, metric in self.val_metrics.items()
                }
            ),
        )

    def forward(self, x: Tensor) -> Output:  # pylint: disable=arguments-differ
        x = self.backbone(x)
        x = self.decode_head(x)
        return x

    def training_step(self, batch: TrainInput) -> Tensor:  # pylint: disable=arguments-differ
        inputs, targets, _ = batch
        inputs_shape = inputs.shape
        inputs = inputs.view(inputs_shape[0] * inputs_shape[1], *inputs_shape[2:])
        outputs = cast(torch.Tensor, self(inputs))
        outputs = outputs.view(inputs_shape[0], inputs_shape[1], *outputs.shape[1:])

        loss = self.criterion(outputs.max(dim=1)[0], targets)
        self.log("train/loss", loss, on_step=True, prog_bar=True)

        return loss

    def validation_step(self, batch: TestInput) -> None:  # pylint: disable=arguments-differ
        inputs, targets, metadata = batch
        outputs = self(inputs)

        loss = self.criterion(outputs, targets)
        self.log("validation/loss", loss, on_epoch=True, prog_bar=True)

        self.update_metrics(self.val_metrics, outputs, targets, metadata)
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
        return Adam(self.parameters(), lr=self.lr)

    def log_dict(self, dictionary: MetricCollection, *args, **kwargs) -> None:
        for name, result in dictionary.compute().items():
            result = cast(Tensor, result)
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
                    outputs.squeeze(-1) if "max1" in name else outputs,
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
