from copy import deepcopy

from lightning import LightningModule
from torch import Tensor, nn
from torch.optim.adam import Adam
from torch.optim.optimizer import Optimizer
from torchmetrics import MetricCollection
from torchmetrics.classification import (
    BinaryAccuracy,
    BinaryAUROC,
    BinaryCohenKappa,
    BinaryPrecision,
    BinaryRecall,
    BinarySpecificity,
)

from ulcerative_colitis.modeling import MLP
from ulcerative_colitis.typing import (
    Output,
    SlideEmbeddingsInput,
    SlideEmbeddingsPredictInput,
)


class UlcerativeColitisModelSlideEmbeddings(LightningModule):
    def __init__(self, lr: float | None = None) -> None:
        super().__init__()
        self.classifier = MLP(768, 256, 128, 1)
        self.criterion = nn.BCELoss()
        self.lr = lr

        metrics = {
            "AUC": BinaryAUROC(),
            "accuracy": BinaryAccuracy(),
            "precision": BinaryPrecision(),
            "recall": BinaryRecall(),
            "specificity": BinarySpecificity(),
            "kappa": BinaryCohenKappa(),
        }

        self.train_metrics = MetricCollection(deepcopy(metrics), prefix="train/")
        self.val_metrics = MetricCollection(deepcopy(metrics), prefix="validation/")
        self.test_metrics = MetricCollection(deepcopy(metrics), prefix="test/")

    def forward(self, x: Tensor) -> Output:
        x = self.classifier(x)
        x = x.sigmoid()

        return x

    def training_step(self, batch: SlideEmbeddingsInput) -> Tensor:  # pylint: disable=arguments-differ
        x, labels, _ = batch
        outputs = self(x)

        loss = self.criterion(outputs, labels)
        self.log("train/loss", loss, on_step=True, prog_bar=True)

        self.train_metrics.update(outputs, labels)
        self.log_dict(self.train_metrics, on_epoch=True, on_step=False)

        return loss

    def validation_step(self, batch: SlideEmbeddingsInput) -> None:  # pylint: disable=arguments-differ
        x, labels, _ = batch

        outputs = self(x)
        loss = self.criterion(outputs, labels)
        self.log("validation/loss", loss, prog_bar=True)

        self.val_metrics.update(outputs, labels)
        self.log_dict(self.val_metrics, on_epoch=True, on_step=False)

    def test_step(self, batch: SlideEmbeddingsInput) -> None:  # pylint: disable=arguments-differ
        x, labels, _ = batch

        outputs = self(x)

        self.test_metrics.update(outputs, labels)
        self.log_dict(self.test_metrics, on_epoch=True, on_step=False)

    def predict_step(self, batch: SlideEmbeddingsPredictInput) -> Output:  # pylint: disable=arguments-differ
        return self(batch[0])

    def configure_optimizers(self) -> Optimizer:
        if self.lr is None:
            raise ValueError("Learning rate must be set for training.")
        return Adam(self.parameters(), lr=self.lr)
