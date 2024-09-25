from lightning import LightningModule
from rationai.mlkit.metrics import LazyMetricDict
from torch import Tensor, nn
from torch.optim.adam import Adam
from torch.optim.optimizer import Optimizer
from torchmetrics import AUROC, Accuracy, MetricCollection, Precision, Recall

from ulcerative_colitis.modeling import ClassificationHead
from ulcerative_colitis.typing import Input, Outputs


class UlcerativeColitisModel(LightningModule):
    def __init__(self, backbone: nn.Module) -> None:
        super().__init__()
        self.backbone = backbone
        self.decode_head = ClassificationHead()
        self.criterion = nn.CrossEntropyLoss()

        self.val_metrics = MetricCollection(
            {
                "AUC": AUROC("multiclass", num_classes=5),
                "accuracy": Accuracy("multiclass", num_classes=5),
                "precision": Precision("multiclass", num_classes=5),
                "recall": Recall("multiclass", num_classes=5),
            },
            prefix="validation/",
        )

        self.test_mterics = LazyMetricDict(self.val_metrics.clone(prefix="test/"))

    def forward(self, x: Input) -> Outputs:
        features = self.backbone(x)
        return self.decode_head(features)

    def training_step(self, batch: Input) -> Tensor:
        inputs, targets, _ = batch
        outputs = self(inputs)

        loss = self.criterion(outputs, targets)
        self.log("train/loss", loss, on_step=True, prog_bar=True)

        return loss

    def validation_step(self, batch: Input) -> None:
        inputs, targets, _ = batch
        outputs = self(inputs)

        loss = self.criterion(outputs, targets)
        self.log("validation/loss", loss, on_epoch=True, prog_bar=True)

        self.val_metrics.update(outputs, targets.argmax(dim=1))
        self.log_dict(self.val_metrics, on_epoch=True)

    def test_step(self, batch: Input) -> None:
        inputs, targets, metadata = batch
        outputs = self(inputs)
        for output, target, slide in zip(
            outputs, targets, metadata["slide"], strict=False
        ):
            self.test_metrics.update(output, target.argmax(dim=1), key=slide)
        self.log_dict(self.test_metrics, on_epoch=True)

    def configure_optimizers(self) -> Optimizer:
        return Adam(self.parameters(), lr=0.0001)
