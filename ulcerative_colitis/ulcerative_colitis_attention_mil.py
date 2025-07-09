from copy import deepcopy
from typing import cast

import torch
from lightning import LightningModule
from rationai.mlkit.metrics import AggregatedMetricCollection
from rationai.mlkit.metrics.aggregators import MaxAggregator
from torch import Tensor, nn
from torch.optim.adam import Adam
from torch.optim.optimizer import Optimizer
from torchmetrics import MetricCollection
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

        metrics = {
            "AUC": BinaryAUROC(),
            "accuracy": BinaryAccuracy(),
            "precision": BinaryPrecision(),
            "recall": BinaryRecall(),
            "specificity": BinarySpecificity(),
        }

        self.train_metrics = MetricCollection(deepcopy(metrics), prefix="train/")
        self.val_metrics = MetricCollection(deepcopy(metrics), prefix="validation/")
        self.test_metrics = MetricCollection(deepcopy(metrics), prefix="test/")

        self.train_agg_metrics = AggregatedMetricCollection(
            deepcopy(metrics), aggregator=MaxAggregator(), prefix="train/agg/"
        )
        self.val_agg_metrics = AggregatedMetricCollection(
            deepcopy(metrics), aggregator=MaxAggregator(), prefix="validation/agg/"
        )
        self.test_agg_metrics = AggregatedMetricCollection(
            deepcopy(metrics), aggregator=MaxAggregator(), prefix="test/agg/"
        )

    def forward(
        self, x: Tensor, return_attention: bool = False
    ) -> Output | tuple[Output, Tensor]:  # pylint: disable=arguments-differ
        x = self.encoder(x)
        attention_weights = sigmoid_normalization(self.attention(x))
        x = self.classifier(x)
        x = x.sigmoid()
        x = torch.sum(attention_weights * x, dim=0)

        if return_attention:
            return x.squeeze(), attention_weights.squeeze()

        return x.squeeze()

    def training_step(self, batch: MILInput) -> Tensor:  # pylint: disable=arguments-differ
        bags, labels, metadatas = batch

        loss = torch.tensor(0.0, device=self.device)
        outputs = []
        for bag, label in zip(bags, labels, strict=True):
            output = self(bag)
            loss += self.criterion(output, label)
            outputs.append(output)

        loss /= len(bags)
        self.log("train/loss", loss, on_step=True, prog_bar=True)

        self.train_metrics.update(torch.tensor(outputs), torch.tensor(labels))
        self.train_agg_metrics.update(
            torch.tensor(outputs),
            torch.tensor(labels),
            [metadata["slide"] for metadata in metadatas],
        )
        self.log_dict(self.train_metrics, on_epoch=True, on_step=False)
        self.log_dict(self.train_agg_metrics, on_epoch=True, on_step=False)

        return loss

    def validation_step(self, batch: MILInput) -> None:  # pylint: disable=arguments-differ
        bags, labels, metadatas = batch

        loss = torch.tensor(0.0, device=self.device)
        outputs = []
        for bag, label in zip(bags, labels, strict=True):
            output = self(bag)
            loss += self.criterion(output, label)
            outputs.append(output)

        loss /= len(bags)
        self.log("validation/loss", loss, prog_bar=True)

        self.val_metrics.update(torch.tensor(outputs), torch.tensor(labels))
        self.val_agg_metrics.update(
            torch.tensor(outputs),
            torch.tensor(labels),
            [metadata["slide"] for metadata in metadatas],
        )
        self.log_dict(self.val_metrics)
        self.log_dict(self.val_agg_metrics)

    def test_step(self, batch: MILInput) -> None:  # pylint: disable=arguments-differ
        bags, labels, metadatas = batch

        outputs = []
        for bag in bags:
            output = self(bag)
            outputs.append(output)

        self.test_metrics.update(torch.tensor(outputs), torch.tensor(labels))
        self.test_agg_metrics.update(
            torch.tensor(outputs),
            torch.tensor(labels),
            [metadata["slide"] for metadata in metadatas],
        )
        self.log_dict(self.test_metrics)
        self.log_dict(self.test_agg_metrics)

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

    def log_dict(self, dictionary: MetricCollection, *args, **kwargs) -> None:
        for name, result in dictionary.compute().items():
            result = cast("Tensor", result)
            if result.shape:
                for i, value in enumerate(result):
                    self.log(f"{name}/{i}", value, *args, **kwargs)
            else:
                self.log(name, result, *args, **kwargs)
