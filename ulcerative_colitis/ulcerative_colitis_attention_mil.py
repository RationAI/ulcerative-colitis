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

from ulcerative_colitis.modeling import sigmoid_normalization
from ulcerative_colitis.typing import MILInput, MILPredictInput, Output


class UlcerativeColitisModelAttentionMIL(LightningModule):
    def __init__(self, lr: float | None = None, alpha: float = 0.01) -> None:
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
        self.alpha = alpha

        metrics: dict[str, Metric] = {
            "AUC": BinaryAUROC(),
            "accuracy": BinaryAccuracy(),
            "precision": BinaryPrecision(),
            "recall": BinaryRecall(),
            "specificity": BinarySpecificity(),
        }

        self.val_metrics = MetricCollection(deepcopy(metrics), prefix="validation/")

        self.test_metrics = MetricCollection(deepcopy(metrics), prefix="test/")

    def forward(
        self, x: Tensor, return_attention: bool = False
    ) -> Output | tuple[Output, Tensor]:  # pylint: disable=arguments-differ
        x = self.encoder(x)
        # attention_weights = torch.softmax(self.attention(x), dim=0)
        attention_weights = sigmoid_normalization(self.attention(x))
        x = torch.sum(attention_weights * x, dim=0)
        x = self.classifier(x)
        x = x.sigmoid()

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
                f"{stage}/attention/coverage_count_{treshold}",
                count,
                on_step=True,
                on_epoch=True,
                prog_bar=True,
            )

    def training_step(self, batch: MILInput) -> Tensor:  # pylint: disable=arguments-differ
        bags, labels, _ = batch

        loss = torch.tensor(0.0, device=self.device)
        for bag, label in zip(bags, labels, strict=True):
            output, attention = self(bag, return_attention=True)
            self.log_attention_coverage(attention, "train")
            l_classification = self.criterion(output, label)
            # l_attention = attention_entropy_loss(attention)
            loss += l_classification  # + self.alpha * l_attention

        loss /= len(bags)
        self.log("train/loss", loss, on_step=True, prog_bar=True)

        return loss

    def validation_step(self, batch: MILInput) -> None:  # pylint: disable=arguments-differ
        bags, labels, _ = batch

        loss = torch.tensor(0.0, device=self.device)
        outputs = []
        for bag, label in zip(bags, labels, strict=True):
            output, attention = self(bag, return_attention=True)
            self.log_attention_coverage(attention, "validation")
            l_classification = self.criterion(output, label)
            # l_attention = attention_entropy_loss(attention)
            loss += l_classification  # + self.alpha * l_attention
            outputs.append(output)

        loss /= len(bags)
        self.log("validation/loss", loss, prog_bar=True)

        self.val_metrics.update(torch.tensor(outputs), torch.tensor(labels))
        self.log_dict(self.val_metrics)

    def test_step(self, batch: MILInput) -> None:  # pylint: disable=arguments-differ
        bags, labels, _ = batch

        outputs = []
        for bag in bags:
            output, attention = self(bag, return_attention=True)
            self.log_attention_coverage(attention, "test")
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
