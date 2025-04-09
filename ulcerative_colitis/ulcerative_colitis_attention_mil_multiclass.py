from copy import deepcopy
from typing import cast

import torch
from lightning import LightningModule
from torch import Tensor, nn
from torch.optim.adam import Adam
from torch.optim.optimizer import Optimizer
from torchmetrics import Metric, MetricCollection
from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassAUROC,
    MulticlassPrecision,
    MulticlassRecall,
    MulticlassSpecificity,
)

from ulcerative_colitis.modeling import sigmoid_normalization
from ulcerative_colitis.typing import MILInput, MILPredictInput, Output


class UlcerativeColitisModelAttentionMILMulticlass(LightningModule):
    def __init__(self, lr: float | None = None) -> None:
        super().__init__()
        self.encoder = nn.Identity()
        self.attention = nn.Sequential(
            nn.Linear(1536, 512),
            nn.Tanh(),
            nn.Linear(512, 1),
        )
        self.classifier = nn.Linear(1536, 3)
        self.criterion = nn.CrossEntropyLoss()
        self.lr = lr

        metrics: dict[str, Metric] = {
            "AUC": MulticlassAUROC(3, average="none"),
            "accuracy": MulticlassAccuracy(3),
            "precision": MulticlassPrecision(3, average="none"),
            "recall": MulticlassRecall(3, average="none"),
            "specificity": MulticlassSpecificity(3, average="none"),
        }

        self.train_metrics = MetricCollection(deepcopy(metrics), prefix="train/")
        self.val_metrics = MetricCollection(deepcopy(metrics), prefix="validation/")
        self.test_metrics = MetricCollection(deepcopy(metrics), prefix="test/")

    def forward(
        self, x: Tensor, return_attention: bool = False
    ) -> Output | tuple[Output, Tensor]:  # pylint: disable=arguments-differ
        x = self.encoder(x)
        attention_weights = sigmoid_normalization(self.attention(x))
        x = self.classifier(x)
        x = torch.softmax(x, dim=0)
        x = torch.sum(attention_weights * x, dim=0)

        if return_attention:
            return x.squeeze(), attention_weights.squeeze()

        return x.squeeze()

    def log_attention_coverage(self, attention_weights: Tensor, stage: str) -> None:
        # Sort attention weights in descending order
        sorted_weights, _ = torch.sort(attention_weights, descending=True)

        tresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        for treshold in tresholds:
            # Find the minimum number of weights that sum to treshold
            cumulative_sum = 0.0
            count = 0
            for weight in sorted_weights:
                cumulative_sum += weight.item()
                count += 1
                if cumulative_sum >= treshold:
                    break

            # Log the result
            self.log(
                f"{stage}/attention/coverage_fraction_{treshold}",
                count / len(attention_weights),
                on_epoch=True,
                prog_bar=True,
            )

    def training_step(self, batch: MILInput) -> Tensor:  # pylint: disable=arguments-differ
        bags, labels, _ = batch

        loss = torch.tensor(0.0, device=self.device)
        outputs = []
        for bag, label in zip(bags, labels, strict=True):
            output = self(bag, return_attention=False)
            loss += self.criterion(output, label)
            outputs.append(output)

        loss /= len(bags)
        self.log("train/loss", loss, on_step=True, prog_bar=True)

        self.train_metrics.update(torch.tensor(outputs), torch.tensor(labels))
        self.log_dict(self.train_metrics, on_epoch=True)

        return loss

    def validation_step(self, batch: MILInput) -> None:  # pylint: disable=arguments-differ
        bags, labels, _ = batch

        loss = torch.tensor(0.0, device=self.device)
        outputs = []
        for bag, label in zip(bags, labels, strict=True):
            output, attention = self(bag, return_attention=True)
            self.log_attention_coverage(attention, "validation")
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

    def log_dict(self, metrics: MetricCollection, *args, **kwargs) -> None:
        for name, result in metrics.compute().items():
            result = cast("Tensor", result)
            if result.shape:
                for i, value in enumerate(result):
                    self.log(f"{name}/{i + 2}", value, *args, **kwargs)
            else:
                self.log(name, result, *args, **kwargs)
