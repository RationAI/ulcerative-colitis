from collections.abc import Iterable

import torch
from hydra.utils import instantiate
from lightning import LightningDataModule
from omegaconf import DictConfig
from sklearn.model_selection import KFold
from torch import Tensor
from torch.utils.data import DataLoader

from ulcerative_colitis.data.datasets.embeddings import EmbeddingsSubset
from ulcerative_colitis.data.samplers import AutoWeightedRandomSampler
from ulcerative_colitis.typing import MetadataMIL, MILInput, MILPredictInput


class DataModule(LightningDataModule):
    def __init__(
        self,
        batch_size: int,
        target_column: str | None = None,
        num_workers: int = 0,
        kfold_splits: int | None = None,
        k: int | None = None,
        **datasets: DictConfig,
    ) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.target_column = target_column
        self.num_workers = num_workers
        self.kfold_splits = kfold_splits
        self.k = k

        if self.kfold_splits is None and self.k is not None:
            raise ValueError("kfold_splits cannot be None if k is set.")

        self.datasets = datasets

        self.setup("")  # fix W0201 [attribute-defined-outside-init]

    def setup(self, stage: str) -> None:
        match stage:
            case "fit" | "validatie":
                assert self.kfold_splits is not None and self.k is not None
                dataset = instantiate(self.datasets["train"])
                kf = KFold(n_splits=self.kfold_splits, random_state=42, shuffle=True)
                train_idx, val_idx = list(kf.split(range(len(dataset))))[self.k - 1]
                self.train = EmbeddingsSubset(dataset, train_idx)
                self.val = EmbeddingsSubset(dataset, val_idx)
            case "test":
                self.test = instantiate(self.datasets["test"])
            case "predict":
                self.predict = instantiate(self.datasets["predict"])

    def train_dataloader(self) -> Iterable[MILInput]:
        if self.target_column is None:
            raise ValueError("target_column must be provided for training")

        return DataLoader(
            self.train,
            batch_size=self.batch_size,
            sampler=AutoWeightedRandomSampler(self.train, self.target_column),
            drop_last=True,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self) -> Iterable[MILInput]:
        return DataLoader(
            self.val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
            persistent_workers=self.num_workers > 0,
        )

    def test_dataloader(self) -> Iterable[MILInput]:
        return DataLoader(
            self.test,
            batch_size=self.batch_size,
            collate_fn=collate_fn,
            num_workers=self.num_workers,
        )

    def predict_dataloader(self) -> Iterable[MILPredictInput]:
        return DataLoader(
            self.predict,
            batch_size=self.batch_size,
            collate_fn=collate_fn_predict,
            num_workers=self.num_workers,
        )


def collate_fn(batch: list[tuple[Tensor, Tensor, MetadataMIL]]) -> MILInput:
    bags = []
    labels = []
    metadatas = []
    for bag, label, metadata in batch:
        bags.append(bag)
        labels.append(label)
        metadatas.append(metadata)
    bags = torch.stack(bags)
    labels = torch.stack(labels)
    return bags, labels, metadatas


def collate_fn_predict(batch: list[tuple[Tensor, MetadataMIL]]) -> MILPredictInput:
    bags = []
    metadatas = []
    for bag, metadata in batch:
        bags.append(bag)
        metadatas.append(metadata)
    bags = torch.stack(bags)
    return bags, metadatas
