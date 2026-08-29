from typing import Any

import torch
from hydra.utils import instantiate
from lightning import LightningDataModule
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import _collate_fn_t

from ml.data.samplers import AutoWeightedRandomSampler
from ml.typing import (
    BagsInput,
    BagsPredictInput,
    EmbeddingsInput,
    EmbeddingsPredictInput,
    Input,
    PredictInput,
    PredictSample,
    Sample,
    TilesInput,
    TilesPredictInput,
)


class DataModule(LightningDataModule):
    def __init__(
        self,
        batch_size: int,
        num_workers: int = 0,
        weighted_sampling: bool = True,
        collate_fn: _collate_fn_t[Sample] | None = None,
        collate_fn_predict: _collate_fn_t[PredictSample] | None = None,
        **datasets: DictConfig,
    ) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.weighted_sampling = weighted_sampling
        self.collate_fn = collate_fn
        self.collate_fn_predict = collate_fn_predict
        self.datasets = datasets

    def setup(self, stage: str) -> None:
        match stage:
            case "fit":
                self.train = instantiate(self.datasets["train"])
                self.val = instantiate(self.datasets["val"])
            case "validate":
                self.val = instantiate(self.datasets["val"])
            case "test":
                self.test = instantiate(self.datasets["test"])
            case "predict":
                self.predict = instantiate(self.datasets["predict"])

    def train_dataloader(self) -> DataLoader[TilesInput | EmbeddingsInput | BagsInput]:
        sampler = (
            AutoWeightedRandomSampler(self.train) if self.weighted_sampling else None
        )
        return DataLoader(
            self.train,
            batch_size=self.batch_size,
            sampler=sampler,
            shuffle=sampler is None,
            drop_last=True,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self) -> DataLoader[TilesInput | EmbeddingsInput | BagsInput]:
        return DataLoader(
            self.val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            persistent_workers=self.num_workers > 0,
        )

    def test_dataloader(self) -> DataLoader[TilesInput | EmbeddingsInput | BagsInput]:
        return DataLoader(
            self.test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
        )

    def predict_dataloader(
        self,
    ) -> DataLoader[TilesPredictInput | EmbeddingsPredictInput | BagsPredictInput]:
        return DataLoader(
            self.predict,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn_predict,
        )


class BagsDataModule(DataModule):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(
            collate_fn=collate_fn,
            collate_fn_predict=collate_fn_predict,
            **kwargs,
        )


def collate_fn(
    batch: list[Sample],
) -> Input:
    inputs, labels, metadatas = zip(*batch, strict=True)
    return torch.stack(list(inputs)), torch.stack(list(labels)), list(metadatas)


def collate_fn_predict(
    batch: list[PredictSample],
) -> PredictInput:
    inputs, metadatas = zip(*batch, strict=True)
    return torch.stack(list(inputs)), list(metadatas)
