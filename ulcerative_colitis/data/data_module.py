from collections.abc import Iterable

import torch
from hydra.utils import instantiate
from lightning import LightningDataModule
from omegaconf import DictConfig
from sklearn.model_selection import KFold
from torch import Tensor
from torch.utils.data import DataLoader

from ulcerative_colitis.data.datasets import create_subset
from ulcerative_colitis.data.datasets.labels import get_target_column
from ulcerative_colitis.data.samplers import AutoWeightedRandomSampler
from ulcerative_colitis.typing import (
    Metadata,
    TileEmbeddingsInput,
    TileEmbeddingsPredictInput,
)


class DataModule(LightningDataModule):
    def __init__(
        self,
        batch_size: int,
        num_workers: int = 0,
        kfold_splits: int | None = None,
        k: int | None = None,
        **datasets: DictConfig,
    ) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.kfold_splits = kfold_splits
        self.k = k

        print("K", self.k)
        print("K-Fold splits", self.kfold_splits)
        if self.kfold_splits is None and self.k is not None:
            raise ValueError("kfold_splits cannot be None if k is set.")

        self.datasets = datasets

        self.setup("")  # fix W0201 [attribute-defined-outside-init]

    def setup(self, stage: str) -> None:
        match stage:
            case "fit" | "validate":
                assert self.kfold_splits is not None and self.k is not None
                dataset = instantiate(self.datasets["train"])
                if self.datasets.get("val") is not None:
                    self.val = instantiate(self.datasets["val"])
                else:
                    kf = KFold(
                        n_splits=self.kfold_splits, random_state=42, shuffle=True
                    )
                    train_idx, val_idx = list(kf.split(range(len(dataset))))[self.k - 1]
                    self.train = create_subset(dataset, train_idx)
                    self.val = create_subset(dataset, val_idx)
            case "test":
                self.test = instantiate(self.datasets["test"])
            case "predict":
                self.predict = instantiate(self.datasets["predict"])

    def train_dataloader(self) -> Iterable[TileEmbeddingsInput]:
        if self.train.mode is None:
            raise ValueError("Dataset mode must be set for training")

        return DataLoader(
            self.train,
            batch_size=self.batch_size,
            sampler=AutoWeightedRandomSampler(
                self.train, get_target_column(self.train.mode)
            ),
            drop_last=True,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self) -> Iterable[TileEmbeddingsInput]:
        return DataLoader(
            self.val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
            persistent_workers=self.num_workers > 0,
        )

    def test_dataloader(self) -> Iterable[TileEmbeddingsInput]:
        return DataLoader(
            self.test,
            batch_size=self.batch_size,
            collate_fn=collate_fn,
            num_workers=self.num_workers,
        )

    def predict_dataloader(self) -> Iterable[TileEmbeddingsPredictInput]:
        return DataLoader(
            self.predict,
            batch_size=self.batch_size,
            collate_fn=collate_fn_predict,
            num_workers=self.num_workers,
        )


def collate_fn(
    batch: list[tuple[Tensor, Tensor, Metadata]],
) -> tuple[Tensor, Tensor, list[Metadata]]:
    inputs = []
    labels = []
    metadatas = []
    for input, label, metadata in batch:
        inputs.append(input)
        labels.append(label)
        metadatas.append(metadata)
    inputs = torch.stack(inputs)
    labels = torch.stack(labels)
    return inputs, labels, metadatas


def collate_fn_predict(
    batch: list[tuple[Tensor, Metadata]],
) -> tuple[Tensor, list[Metadata]]:
    inputs = []
    metadatas = []
    for input, metadata in batch:
        inputs.append(input)
        metadatas.append(metadata)
    inputs = torch.stack(inputs)
    return inputs, metadatas
