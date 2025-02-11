from collections.abc import Iterable

from hydra.utils import instantiate
from lightning import LightningDataModule
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from ulcerative_colitis.data.samplers import AutoWeightedRandomSampler
from ulcerative_colitis.typing import PredictInput, TestInput, TrainInput


class DataModule(LightningDataModule):
    def __init__(
        self,
        batch_size: int,
        target_column: str | None = None,
        num_workers: int = 0,
        **datasets: DictConfig,
    ) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.target_column = target_column
        self.num_workers = num_workers
        self.datasets = datasets

        self.setup("")  # fix W0201 [attribute-defined-outside-init]

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

    def train_dataloader(self) -> Iterable[TrainInput]:
        if self.target_column is None:
            raise ValueError("target_column must be provided for training")

        return DataLoader(
            self.train,
            batch_size=self.batch_size,
            sampler=AutoWeightedRandomSampler(self.train, self.target_column),
            drop_last=True,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self) -> Iterable[TestInput]:
        return DataLoader(
            self.val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
        )

    def test_dataloader(self) -> Iterable[TestInput]:
        return DataLoader(
            self.test, batch_size=self.batch_size, num_workers=self.num_workers
        )

    def predict_dataloader(self) -> list[Iterable[PredictInput]]:
        return [
            DataLoader(
                dataset, batch_size=self.batch_size, num_workers=self.num_workers
            )
            for dataset in self.predict.datasets
        ]
