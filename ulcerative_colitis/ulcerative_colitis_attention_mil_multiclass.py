from copy import deepcopy
from typing import cast

import torch
from lightning import LightningModule
from torch import Tensor, nn
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

from ulcerative_colitis.modeling import sigmoid_normalization
from ulcerative_colitis.typing import (
    Output,
    TileEmbeddingsInput,
    TileEmbeddingsPredictInput,
)


class UlcerativeColitisModelAttentionMILMulticlass(LightningModule):
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
        self.criterion = nn.CrossEntropyLoss()
        self.lr = lr

        metrics = {
            "AUC": MulticlassAUROC(num_classes, average="none"),
            "accuracy": MulticlassAccuracy(num_classes),
            "precision": MulticlassPrecision(num_classes, average="none"),
            "recall": MulticlassRecall(num_classes, average="none"),
            "specificity": MulticlassSpecificity(num_classes, average="none"),
            "kappa": MulticlassCohenKappa(num_classes),
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

    def training_step(self, batch: TileEmbeddingsInput) -> Tensor:  # pylint: disable=arguments-differ
        bags, labels, _ = batch

        outputs = self(bags)
        loss = self.criterion(outputs, labels)
        self.log("train/loss", loss, on_step=True, prog_bar=True, batch_size=len(bags))

        self.train_metrics.update(torch.softmax(outputs, dim=-1), labels)
        self.log_dict(
            self.train_metrics, on_epoch=True, on_step=False, batch_size=len(bags)
        )

        return loss

    def validation_step(self, batch: TileEmbeddingsInput) -> None:  # pylint: disable=arguments-differ
        bags, labels, _ = batch

        outputs = self(bags)
        loss = self.criterion(outputs, labels)
        self.log("validation/loss", loss, prog_bar=True, batch_size=len(bags))

        self.val_metrics.update(torch.softmax(outputs, dim=-1), labels)
        self.log_dict(
            self.val_metrics, on_epoch=True, on_step=False, batch_size=len(bags)
        )

    def test_step(self, batch: TileEmbeddingsInput) -> None:  # pylint: disable=arguments-differ
        bags, labels, _ = batch

        outputs = self(bags)

        self.test_metrics.update(torch.softmax(outputs, dim=-1), labels)
        self.log_dict(
            self.test_metrics, on_epoch=True, on_step=False, batch_size=len(bags)
        )

        return torch.softmax(outputs, dim=-1)

    def predict_step(  # pylint: disable=arguments-differ
        self, batch: TileEmbeddingsPredictInput, batch_idx: int, dataloader_idx: int = 0
    ) -> Output:
        return torch.softmax(self(batch[0]), dim=-1)

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
