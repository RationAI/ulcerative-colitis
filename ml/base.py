from abc import abstractmethod
from collections.abc import Mapping
from typing import cast

from lightning import LightningModule
from rationai.mlkit.metrics import (
    AggregatedMetricCollection,
    MaxAggregator,
    MeanAggregator,
)
from torch import Tensor
from torch.nn import Module, ModuleDict, Softmax
from torch.optim.adam import Adam
from torch.optim.optimizer import Optimizer
from torchmetrics import MetricCollection
from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassAUROC,
    MulticlassCohenKappa,
    MulticlassPrecision,
    MulticlassRecall,
    MulticlassSpecificity,
)

from ml.typing import MetadataTilesBatch, Output, TilesInput, TilesPredictInput


class BaseModule(LightningModule):
    criterion: Module

    def __init__(self, backbone: Module, lr: float) -> None:
        super().__init__()
        self.backbone = backbone
        self.lr = lr
        self.activation = Softmax(dim=-1)

    N_CLASSES = 5

    def _build_metrics(self) -> None:
        metrics = {
            "AUC": MulticlassAUROC(num_classes=self.N_CLASSES, average=None),
            "accuracy": MulticlassAccuracy(num_classes=self.N_CLASSES),
            "precision": MulticlassPrecision(num_classes=self.N_CLASSES, average=None),
            "recall": MulticlassRecall(num_classes=self.N_CLASSES, average=None),
            "specificity": MulticlassSpecificity(
                num_classes=self.N_CLASSES, average=None
            ),
            "kappa": MulticlassCohenKappa(num_classes=self.N_CLASSES),
        }
        val_metrics: dict[str, MetricCollection] = {
            "tiles_all": MetricCollection(metrics, prefix="validation/tiles/"),
            "slide_max": AggregatedMetricCollection(
                metrics,
                aggregator=MaxAggregator(),
                prefix="validation/scenes/max/",
            ),
            "slide_mean": AggregatedMetricCollection(
                metrics,
                aggregator=MeanAggregator(),
                prefix="validation/slides/mean/",
            ),
        }
        test_metrics = {
            name: metric.clone(
                prefix=cast("str", metric.prefix).replace("validation", "test")
            )
            for name, metric in val_metrics.items()
        }
        self.val_metrics = cast("dict[str, MetricCollection]", ModuleDict(val_metrics))
        self.test_metrics = cast(
            "dict[str, MetricCollection]", ModuleDict(test_metrics)
        )

    @abstractmethod
    def forward(self, x: Tensor) -> Output: ...

    def training_step(self, batch: TilesInput) -> Tensor:
        inputs, targets, _ = batch
        outputs = self(inputs)
        loss = self.criterion(outputs, targets)
        self.log("train/loss", loss, on_step=True, prog_bar=True)
        return loss

    def validation_step(self, batch: TilesInput) -> None:
        inputs, targets, metadata = batch
        outputs = self(inputs)
        loss = self.criterion(outputs, targets)
        self.log("validation/loss", loss, on_epoch=True, prog_bar=True)
        targets = targets.reshape(-1)
        probabilities = self.activation(outputs)
        self.update_metrics(self.val_metrics, probabilities, targets, metadata)
        self.log_metrics(self.val_metrics)

    def test_step(self, batch: TilesInput) -> Output:
        inputs, targets, metadata = batch
        outputs = self(inputs)
        targets = targets.reshape(-1)
        probabilities = self.activation(outputs)
        self.update_metrics(self.test_metrics, probabilities, targets, metadata)
        self.log_metrics(self.test_metrics)
        return probabilities

    def predict_step(self, batch: TilesPredictInput) -> Output:
        return self(batch[0])

    def configure_optimizers(self) -> Optimizer:
        return Adam(self.parameters(), lr=self.lr)

    def log_dict(self, dictionary: MetricCollection, *args, **kwargs) -> None:
        for name, metric in dictionary.items():
            result = cast("Tensor", metric.compute())
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
        metadata: MetadataTilesBatch,
    ) -> None:
        for name, metric in metrics.items():
            if "slide" in name:
                metric.update(outputs, targets, key=metadata["slide_name"])
            else:
                metric.update(outputs, targets)

    def log_metrics(self, metrics: dict[str, MetricCollection]) -> None:
        for metric in metrics.values():
            self.log_dict(metric, on_epoch=True)
