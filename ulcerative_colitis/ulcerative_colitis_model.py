from typing import cast

from lightning import LightningModule
from rationai.mlkit.lightning.loggers import MLFlowLogger
from rationai.mlkit.metrics import LazyMetricDict
from torch import Tensor, nn
from torch.optim.adam import Adam
from torch.optim.optimizer import Optimizer
from torchmetrics import Metric, MetricCollection
from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassAUROC,
    MulticlassPrecision,
    MulticlassRecall,
)

from ulcerative_colitis.loss import CumulativeLinkLoss
from ulcerative_colitis.metrics import MetricAggregator
from ulcerative_colitis.modeling import LogisticCumulativeLink
from ulcerative_colitis.modeling.regression_head import RegressionHead
from ulcerative_colitis.typing import Input, MetadataBatch, Output


class UlcerativeColitisModel(LightningModule):
    def __init__(self, backbone: nn.Module) -> None:
        super().__init__()
        self.backbone = backbone
        self.n_classes = 5
        self.decode_head = RegressionHead()
        self.cumulative_link = LogisticCumulativeLink(num_classes=self.n_classes)
        self.criterion = CumulativeLinkLoss()

        metrics: dict[str, Metric] = {
            "AUC": MulticlassAUROC(num_classes=self.n_classes, average=None),
            "accuracy": MulticlassAccuracy(num_classes=self.n_classes),
            "precision": MulticlassPrecision(num_classes=self.n_classes, average=None),
            "recall": MulticlassRecall(num_classes=self.n_classes, average=None),
        }

        self.val_metrics: dict[str, MetricCollection] = {
            "tiles_all": MetricCollection(metrics, prefix="validation/tiles/"),
            "scene_max": MetricAggregator(
                metrics, agg="max", prefix="validation/scenes_max/"
            ),
            "slide_max": MetricAggregator(
                metrics, agg="max", prefix="validation/slides_max/"
            ),
            "scene_mean": MetricAggregator(
                metrics, agg="mean", prefix="validation/scenes_mean/"
            ),
            "slide_mean": MetricAggregator(
                metrics, agg="mean", prefix="validation/slides_mean/"
            ),
        }

        self.test_metrics: dict[str, MetricCollection] = {
            name: metric.clone(
                prefix=cast(str, metric.prefix).replace("validation", "test")
            )
            for name, metric in self.val_metrics.items()
        }
        self.test_metrics["tiles_scene"] = LazyMetricDict(
            self.test_metrics["tiles_all"].clone(prefix="test/tiles/")
        )
        self.test_metrics["tiles_slide"] = LazyMetricDict(
            self.test_metrics["tiles_all"].clone(prefix="test/tiles/")
        )

        # Register each MetricCollection as an attribute
        for name, metric in self.val_metrics.items():
            setattr(self, f"val_metrics_{name}", metric)
        for name, metric in self.test_metrics.items():
            setattr(self, f"test_metrics_{name}", metric)

    # def to(self, *args, **kwargs) -> Self:
    #     model = super().to(*args, **kwargs)
    #     model.val_metrics = {
    #         name: metric.to(self.device) for name, metric in self.val_metrics.items()
    #     }
    #     model.test_metrics = {
    #         name: metric.to(self.device) for name, metric in self.test_metrics.items()
    #     }
    #     return model

    def forward(self, x: Tensor) -> Output:  # pylint: disable=arguments-differ
        x = self.backbone(x)
        x = self.decode_head(x)
        x = self.cumulative_link(x)
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

        self._update_metrics(self.val_metrics, outputs, targets.reshape(-1), metadata)

    def on_validation_epoch_end(self) -> None:
        self._log_metrics(self.val_metrics)

    def test_step(self, batch: Input) -> None:  # pylint: disable=arguments-differ
        inputs, targets, metadata = batch
        outputs = self(inputs)

        self._update_metrics(self.test_metrics, outputs, targets.reshape(-1), metadata)

    def on_test_epoch_end(self) -> None:
        self._log_metrics(self.test_metrics)

    def predict_step(self, batch: Input) -> Output:  # pylint: disable=arguments-differ
        inputs, _, _ = batch
        return self(inputs)

    def configure_optimizers(self) -> Optimizer:
        return Adam(self.parameters(), lr=0.00001)

    def _update_metrics(
        self,
        metrics: dict[str, MetricCollection],
        outputs: Tensor,
        targets: Tensor,
        metadata: MetadataBatch,
    ) -> None:
        metrics["tiles_all"].update(outputs, targets)
        for output, target, scene in zip(
            outputs, targets, metadata["slide"], strict=True
        ):
            slide = scene.split("_scene_")[0]
            for name, metric in metrics.items():
                if name == "tiles_all":
                    continue
                if "scene" in name:
                    metric.update(output, target, key=scene)
                if "slide" in name:
                    metric.update(output, target, key=slide)

    def _log_metrics(self, metrics: dict[str, MetricCollection]) -> None:
        for name, metric in metrics.items():
            if isinstance(metric, LazyMetricDict):
                assert isinstance(self.logger, MLFlowLogger)
                self.logger.log_table(metric.compute(), f"{name}.parquet")
            else:
                self.log_dict(metric, on_epoch=True)

            metric.reset()
