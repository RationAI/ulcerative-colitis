from copy import deepcopy

import torch
from lightning import LightningModule
from torch import Tensor, nn
from torch.optim.adam import Adam
from torch.optim.optimizer import Optimizer
from torchmetrics import Metric, MetricCollection
from torchmetrics.classification import (
    BinaryAccuracy,
    BinaryAUROC,
    BinaryPrecision,
    BinaryRecall,
    BinarySpecificity,
)

from ulcerative_colitis.typing import MILInput, MILPredictInput, Output


class UlcerativeColitisModelAttentionMIL(LightningModule):
    def __init__(self, lr: float | None = None) -> None:
        super().__init__()
        self.encoder = nn.Identity()
        self.attention = nn.Sequential(
            nn.Linear(1536, 512),
            nn.Tanh(),
            nn.Linear(512, 1),
        )
        self.classifier = nn.Linear(1536, 1)
        self.criterion = nn.BCELoss()
        self.lr = lr

        metrics: dict[str, Metric] = {
            "AUC": BinaryAUROC(),
            "accuracy": BinaryAccuracy(),
            "precision": BinaryPrecision(),
            "recall": BinaryRecall(),
            "specificity": BinarySpecificity(),
        }

        self.val_metrics = MetricCollection(deepcopy(metrics), prefix="validation/")

        self.test_metrics = MetricCollection(deepcopy(metrics), prefix="test/")

    def forward(self, x: Tensor) -> Output:  # pylint: disable=arguments-differ
        x = self.encoder(x)
        attention_weights = torch.softmax(self.attention(x), dim=0)
        x = torch.sum(attention_weights * x, dim=0)
        x = self.classifier(x)
        x = x.sigmoid()
        return x.squeeze()

    def training_step(self, batch: MILInput) -> Tensor:  # pylint: disable=arguments-differ
        bags, labels, _ = batch

        loss = torch.tensor(0.0, device=self.device)
        for bag, label in zip(bags, labels, strict=True):
            output = self(bag)
            loss += self.criterion(output, label)

        loss /= len(bags)
        self.log("train/loss", loss, on_step=True, prog_bar=True)

        return loss

    def validation_step(self, batch: MILInput) -> None:  # pylint: disable=arguments-differ
        bags, labels, _ = batch

        loss = torch.tensor(0.0, device=self.device)
        outputs = []
        for bag, label in zip(bags, labels, strict=True):
            output = self(bag)
            loss += self.criterion(output, label)
            outputs.append(output)

        loss /= len(bags)
        self.log("validation/loss", loss, prog_bar=True)

        self.val_metrics.update(torch.tensor(outputs), torch.tensor(labels))
        self.log_dict(self.val_metrics)

    def test_step(self, batch: MILInput) -> None:  # pylint: disable=arguments-differ
        bags, labels, _ = batch

        outputs = []
        for bag in bags:
            output = self(bag)
            outputs.append(output)

        self.test_metrics.update(torch.tensor(outputs), torch.tensor(labels))
        self.log_dict(self.test_metrics)

    def predict_step(  # pylint: disable=arguments-differ
        self, batch: MILPredictInput, batch_idx: int, dataloader_idx: int = 0
    ) -> Output:
        bags, _ = batch

        outputs = []
        for bag in bags:
            output = self(bag)
            outputs.append(output)

        return torch.tensor(outputs)

    def configure_optimizers(self) -> Optimizer:
        if self.lr is None:
            raise ValueError("Learning rate must be set for training.")
        return Adam(self.parameters(), lr=self.lr)
