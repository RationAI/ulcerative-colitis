from typing import cast

from lightning import LightningModule
from rationai.mlkit.lightning.loggers import MLFlowLogger
from rationai.mlkit.metrics import LazyMetricDict
from torch import Tensor, nn
from torch.optim.adam import Adam
from torch.optim.optimizer import Optimizer
from torchmetrics import AUROC, Accuracy, MetricCollection, Precision, Recall

from ulcerative_colitis.loss import CumulativeLinkLoss
from ulcerative_colitis.metrics import MetricAggregator
from ulcerative_colitis.modeling import LogisticCumulativeLink
from ulcerative_colitis.modeling.regression_head import RegressionHead
from ulcerative_colitis.typing import Input, MetadataBatch, Output


class UlcerativeColitisModel(LightningModule):
    def __init__(self, backbone: nn.Module) -> None:
        super().__init__()
        self.backbone = backbone
        self.num_classes = 5
        self.decode_head = RegressionHead()
        self.cumulative_link = LogisticCumulativeLink(num_classes=self.num_classes)
        self.criterion = CumulativeLinkLoss()

        metrics = {
            "AUC": AUROC("multiclass", num_classes=self.num_classes, average=None),
            "accuracy": Accuracy("multiclass", num_classes=self.num_classes),
            "precision": Precision(
                "multiclass", num_classes=self.num_classes, average=None
            ),
            "recall": Recall("multiclass", num_classes=self.num_classes, average=None),
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

        self._update_metrics(self.val_metrics, outputs, targets, metadata)

    def on_validation_epoch_end(self) -> None:
        self._log_metrics(self.val_metrics)

    def test_step(self, batch: Input) -> None:  # pylint: disable=arguments-differ
        inputs, targets, metadata = batch
        outputs = self(inputs)

        self._update_metrics(self.test_metrics, outputs, targets, metadata)

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
        print("outputs", outputs)
        print("targets", targets)
        metrics["tiles_all"].update(outputs, targets)
        for output, target, scene in zip(
            outputs, targets, metadata["slide"], strict=True
        ):
            slide = scene.split("_scene_")[0]

            metrics["tiles_scene"].update(output, target, key=scene)
            metrics["scene_max"].update(output, target, key=scene)
            metrics["scene_mean"].update(output, target, key=scene)

            metrics["tiles_slide"].update(output, target, key=slide)
            metrics["slide_max"].update(output, target, key=slide)
            metrics["slide_mean"].update(output, target, key=slide)

    def _log_metrics(self, metrics: dict[str, MetricCollection]) -> None:
        for name, metric in metrics.items():
            if isinstance(metric, LazyMetricDict):
                assert isinstance(self.logger, MLFlowLogger)
                self.logger.log_table(metric.compute(), f"{name}.parquet")
            else:
                self.log_dict(metric, on_epoch=True)
