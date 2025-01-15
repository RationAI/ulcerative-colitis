from collections.abc import Mapping
from copy import deepcopy
from typing import cast

from lightning import LightningModule
from rationai.mlkit.lightning.loggers import MLFlowLogger
from rationai.mlkit.metrics import NestedMetricCollection
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
from ulcerative_colitis.typing import Input, MetadataBatch, Output, PredictInput


class UlcerativeColitisModelBinary(LightningModule):
    def __init__(self, backbone: Module) -> None:
        super().__init__()
        self.backbone = backbone
        self.decode_head = BinaryClassificationHead()
        self.criterion = BCELoss()

        metrics: dict[str, Metric] = {
            "AUC": BinaryAUROC(),
            "accuracy": BinaryAccuracy(),
            "precision": BinaryPrecision(),
            "recall": BinaryRecall(),
        }

        self.val_metrics: dict[str, MetricCollection] = cast(
            dict,
            ModuleDict(
                {
                    "tiles_all": MetricCollection(
                        deepcopy(metrics), prefix="validation/tiles/"
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

        self.test_metrics_nested = NestedMetricCollection(
            self.test_metrics["tiles_all"].clone(),
            key_name="slide",
        )

    def forward(self, x: Tensor) -> Output:  # pylint: disable=arguments-differ
        x = self.backbone(x)
        x = self.decode_head(x)
        return x

    def training_step(self, batch: Input) -> Tensor:  # pylint: disable=arguments-differ
        inputs, targets, _ = batch
        outputs = self(inputs)

        loss = self.criterion(outputs, targets)
        self.log("train/loss", loss, on_step=True, prog_bar=True)

        return loss

    def validation_step(self, batch: Input) -> None:  # pylint: disable=arguments-differ
        inputs, targets, metadata = batch
        outputs = self(inputs)

        loss = self.criterion(outputs, targets)
        self.log("validation/loss", loss, on_epoch=True, prog_bar=True)

        self.update_metrics(self.val_metrics, outputs, targets, metadata)
        self.log_metrics(self.val_metrics)

    def test_step(self, batch: Input) -> None:  # pylint: disable=arguments-differ
        inputs, targets, metadata = batch
        outputs = self(inputs)

        self.update_metrics(self.test_metrics, outputs, targets, metadata)
        self.log_metrics(self.test_metrics)

    def on_test_epoch_end(self) -> None:
        for name, metric in self.test_metrics_nested.items():
            assert isinstance(self.logger, MLFlowLogger)
            self.logger.log_table(metric.compute(), f"{name}.json")
            metric.reset()

    def predict_step(self, batch: PredictInput) -> Output:  # pylint: disable=arguments-differ
        inputs, metadata = batch
        outputs = self(inputs)

        table = {
            "slide": metadata["slide"],
            "x": metadata["x"].cpu(),
            "y": metadata["y"].cpu(),
        }
        for i in range(outputs.shape[1]):
            table[f"pred_{i}"] = outputs[:, i].cpu()
        assert isinstance(self.logger, MLFlowLogger)
        self.logger.log_table(table, artifact_file="predictions.parquet")

        return outputs

    def configure_optimizers(self) -> Optimizer:
        return Adam(self.parameters(), lr=0.001)

    def log_dict(self, dictionary: MetricCollection, *args, **kwargs) -> None:
        for name, metric in dictionary.items():
            result = cast(Tensor, metric.compute())
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
                    outputs,
                    targets,
                    keys=metadata["slide"],
                    x=metadata["x"],
                    y=metadata["y"],
                )
            else:
                metric.update(outputs, targets)

    def log_metrics(self, metrics: dict[str, MetricCollection]) -> None:
        for metric in metrics.values():
            self.log_dict(metric, on_epoch=True)
