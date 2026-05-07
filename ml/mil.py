from copy import deepcopy
from typing import cast

import torch
from lightning import LightningModule
from torch import Tensor, nn
from torch.optim.adam import Adam
from torch.optim.optimizer import Optimizer
from torchmetrics import Metric, MetricCollection
from torchmetrics.classification import (
    AUROC,
    Accuracy,
    CohenKappa,
    Precision,
    Recall,
    Specificity,
)

from ml.modeling import sigmoid_normalization
from ml.typing import BagsInput, BagsPredictInput, Output


class MILBase(LightningModule):
    def __init__(
        self, foundation: str, num_classes: int, lr: float | None = None
    ) -> None:
        super().__init__()
        match foundation:
            case "prov-gigapath" | "uni2":
                input_dim = 1536
            case "uni":
                input_dim = 1024
            case "virchow" | "virchow2":
                input_dim = 2560
            case _:
                raise ValueError(f"Unknown foundation model: {foundation}")

        self.encoder = nn.Identity()
        self.attention = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.Tanh(),
            nn.Linear(512, 1),
        )
        self.classifier = nn.Linear(input_dim, num_classes)
        self.criterion = (
            nn.CrossEntropyLoss() if num_classes > 1 else nn.BCEWithLogitsLoss()
        )
        self.activation = nn.Sigmoid() if num_classes == 1 else nn.Softmax(dim=-1)
        self.lr = lr

        task = "binary" if num_classes == 1 else "multiclass"
        metrics: dict[str, Metric | MetricCollection] = {
            "AUC": AUROC(task=task, num_classes=num_classes, average="none"),
            "accuracy": Accuracy(task=task, num_classes=num_classes),
            "precision": Precision(task=task, num_classes=num_classes, average="none"),
            "recall": Recall(task=task, num_classes=num_classes, average="none"),
            "specificity": Specificity(
                task=task, num_classes=num_classes, average="none"
            ),
            "kappa": CohenKappa(task=task, num_classes=num_classes),
        }

        self.train_metrics = MetricCollection(deepcopy(metrics), prefix="train/")
        self.val_metrics = MetricCollection(deepcopy(metrics), prefix="validation/")
        self.test_metrics = MetricCollection(deepcopy(metrics), prefix="test/")

    def forward(self, x: Tensor) -> Output:
        # x has shape (batch_size, num_tiles_padded, embedding_dim)
        x = self.encoder(x)
        attention_weights = sigmoid_normalization(self.attention(x))
        mask = (x.abs() > 1e-6).any(dim=-1, keepdim=True).float()
        attention_weights = attention_weights * mask
        attention_weights = attention_weights / attention_weights.sum(
            dim=1, keepdim=True
        )
        x = self.classifier(x)
        x = torch.sum(attention_weights * x, dim=1)

        return x.squeeze(-1)

    def training_step(self, batch: BagsInput) -> Tensor:
        bags, labels, _ = batch

        outputs = self(bags)
        loss = self.criterion(outputs, labels)
        self.log("train/loss", loss, on_step=True, prog_bar=True, batch_size=len(bags))

        self.train_metrics.update(self.activation(outputs), labels)
        self.log_dict(
            self.train_metrics, on_epoch=True, on_step=False, batch_size=len(bags)
        )

        return loss

    def validation_step(self, batch: BagsInput) -> None:
        bags, labels, _ = batch

        outputs = self(bags)
        loss = self.criterion(outputs, labels)
        self.log("validation/loss", loss, prog_bar=True, batch_size=len(bags))

        self.val_metrics.update(self.activation(outputs), labels)
        self.log_dict(
            self.val_metrics, on_epoch=True, on_step=False, batch_size=len(bags)
        )

    def test_step(self, batch: BagsInput) -> None:
        bags, labels, _ = batch

        outputs = self(bags)

        self.test_metrics.update(self.activation(outputs), labels)
        self.log_dict(
            self.test_metrics, on_epoch=True, on_step=False, batch_size=len(bags)
        )

        return outputs

    def predict_step(self, batch: BagsPredictInput) -> Output:
        return self.activation(self(batch[0]))

    def configure_optimizers(self) -> Optimizer:
        if self.lr is None:
            raise ValueError("Learning rate must be set for training.")
        return Adam(self.parameters(), lr=self.lr)

    def log_dict(self, metrics: MetricCollection, *args, **kwargs) -> None:
        for name, result in metrics.compute().items():
            result = cast("Tensor", result)
            if result.shape:
                for i, value in enumerate(result):
                    self.log(f"{name}/{i}", value, *args, **kwargs)
            else:
                self.log(name, result, *args, **kwargs)
