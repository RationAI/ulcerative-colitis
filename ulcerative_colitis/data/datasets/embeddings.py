from collections.abc import Sequence
from enum import Enum
from pathlib import Path
from typing import Generic, TypeVar, cast

import mlflow
import mlflow.artifacts
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, Subset

from ulcerative_colitis.typing import MetadataMIL, MILPredictSample, MILSample


T = TypeVar("T", bound=MILSample | MILPredictSample)


class EmbeddingsMode(Enum):
    NEUTROPHILS = "neutrophils"
    NANCY_HIGH = "nancy_high"
    NANCY_LOW = "nancy_low"
    NANCY_MIX = "nancy_mix"


class _Embeddings(Dataset[T], Generic[T]):
    def __init__(
        self,
        uri: str,
        uri_embeddings: str,
        mode: EmbeddingsMode | str,
        minimum_region_size: int = 100,
        folder: str | None = None,
        slide_names: list[str] | None = None,
        include_labels: bool = True,
    ) -> None:
        self.mode = EmbeddingsMode(mode)
        artifacts = Path(mlflow.artifacts.download_artifacts(uri))
        self.tiles = pd.read_parquet(artifacts / "tiles.parquet")
        self.slides = pd.read_parquet(artifacts / "slides.parquet")
        self.slides = process_slides(self.slides, self.mode, slide_names)

        if folder is not None:
            self.folder = Path(folder)
        if not self.folder.exists():
            self.folder = Path(mlflow.artifacts.download_artifacts(uri_embeddings))

        self.minimum_region_size = minimum_region_size
        self.include_labels = include_labels
        self.bags = self._create_bags(minimum_region_size)

    def _create_bags(self, minimum_region_size: int) -> pd.DataFrame:
        bags = self.tiles.groupby(["slide_id", "region"]).agg("size")
        return bags[bags >= minimum_region_size].reset_index()

    def __len__(self) -> int:
        return len(self.bags)

    def __getitem__(self, idx: int) -> MILSample | MILPredictSample:
        slide_id = self.bags.iloc[idx]["slide_id"]
        region = self.bags.iloc[idx]["region"]

        slide_metadata = self.slides.query(f"id == {slide_id!s}").iloc[0]
        tiles = self.tiles.query(f"slide_id == {slide_id!s} and region == {region}")
        slide_name = get_slide_name(slide_metadata)
        embeddings = cast(
            "torch.Tensor",
            torch.load(
                (self.folder / slide_name).with_suffix(".pt"), map_location="cpu"
            ),
        )

        metadata = MetadataMIL(
            slide=slide_name,
            slide_path=Path(slide_metadata["path"]),
            level=slide_metadata["level"],
            tile_extent_x=slide_metadata["tile_extent_x"],
            tile_extent_y=slide_metadata["tile_extent_y"],
            tiles=tiles,
            x=torch.from_numpy(tiles["x"].to_numpy()),
            y=torch.from_numpy(tiles["y"].to_numpy()),
        )

        if not self.include_labels:
            return embeddings, metadata

        label = get_label(slide_metadata, self.mode)

        return embeddings, label, metadata


class Embeddings(_Embeddings[MILSample]):
    def __init__(
        self,
        uri: str,
        uri_embeddings: str,
        mode: EmbeddingsMode | str,
        minimum_region_size: int = 100,
        folder: str | None = None,
        slide_names: list[str] | None = None,
    ) -> None:
        super().__init__(
            uri=uri,
            uri_embeddings=uri_embeddings,
            mode=mode,
            minimum_region_size=minimum_region_size,
            folder=folder,
            slide_names=slide_names,
            include_labels=True,
        )


class EmbeddingsPredict(_Embeddings[MILPredictSample]):
    def __init__(
        self,
        uri: str,
        uri_embeddings: str,
        mode: EmbeddingsMode | str,
        minimum_region_size: int = 100,
        folder: str | None = None,
        slide_names: list[str] | None = None,
    ) -> None:
        super().__init__(
            uri=uri,
            uri_embeddings=uri_embeddings,
            mode=mode,
            minimum_region_size=minimum_region_size,
            folder=folder,
            slide_names=slide_names,
            include_labels=False,
        )


class EmbeddingsSubset(Subset[MILSample]):
    def __init__(
        self,
        dataset: Embeddings,
        slide_indices: Sequence[int],
    ) -> None:
        self.slides = dataset.slides.iloc[list(slide_indices)].reset_index()
        bag_indices = np.flatnonzero(dataset.bags["slide_id"].isin(self.slides["id"]))
        super().__init__(dataset, bag_indices.tolist())


def get_slide_name(slide_metadata: pd.Series) -> str:
    return Path(slide_metadata["path"]).stem


def process_slides(
    slides: pd.DataFrame, mode: EmbeddingsMode, slide_names: list[str] | None = None
) -> pd.DataFrame:
    match mode:
        case EmbeddingsMode.NEUTROPHILS:
            slides["neutrophils"] = slides["nancy_index"] >= 2
        case EmbeddingsMode.NANCY_LOW:
            slides = slides[slides["nancy_index"] < 2]
        case EmbeddingsMode.NANCY_HIGH:
            slides = slides[slides["nancy_index"] >= 2]
            slides["ulceration"] = slides["nancy_index"] == 4
            slides["nancy_index"] -= 2
        case EmbeddingsMode.NANCY_MIX:
            slides = slides[slides["nancy_index"] < 4]
            slides["nancy_mix"] = slides["nancy_index"].isin([1, 3])

    if slide_names is not None:
        slides = slides[slides["path"].str.contains("|".join(slide_names))]
    return slides


def get_label(slide_metadata: pd.Series, mode: EmbeddingsMode) -> torch.Tensor:
    match mode:
        case EmbeddingsMode.NEUTROPHILS:
            return torch.tensor(slide_metadata["neutrophils"]).float()
        case EmbeddingsMode.NANCY_LOW:
            return torch.tensor(slide_metadata["nancy_index"]).float()
        case EmbeddingsMode.NANCY_HIGH:
            return torch.tensor(slide_metadata["ulceration"]).float()
            # return torch.tensor(slide_metadata["nancy_index"]).long()
        case EmbeddingsMode.NANCY_MIX:
            return torch.tensor(slide_metadata["nancy_mix"]).float()
