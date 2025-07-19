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


class _Embeddings(Dataset[T], Generic[T]):
    def __init__(
        self,
        uri: str,
        uri_embeddings: str,
        mode: EmbeddingsMode | str,
        minimum_region_size: int = 100,
        folder_embeddings: str | None = None,
        include_labels: bool = True,
    ) -> None:
        self.mode = EmbeddingsMode(mode)
        artifacts = Path(mlflow.artifacts.download_artifacts(uri))
        self.tiles = pd.read_parquet(artifacts / "tiles.parquet")
        self.slides = pd.read_parquet(artifacts / "slides.parquet")
        self.slides = process_slides(self.slides, self.mode)

        if folder_embeddings is not None:
            self.folder_embeddings = Path(folder_embeddings)
        if not self.folder_embeddings.exists():
            self.folder_embeddings = Path(
                mlflow.artifacts.download_artifacts(uri_embeddings)
            )

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
        slide_tiles = self.tiles.query(f"slide_id == {slide_id!s}").reset_index(
            drop=True
        )
        region_tiles = slide_tiles.query(f"region == {region}")
        name = str(slide_metadata["name"])
        embeddings = cast(
            "torch.Tensor",
            torch.load(
                (self.folder_embeddings / f"{name}_region_{region:03d}.pt"),
                map_location="cpu",
            ),
        )

        metadata = MetadataMIL(
            slide=name,
            slide_path=Path(slide_metadata["path"]),
            level=slide_metadata["level"],
            tile_extent_x=slide_metadata["tile_extent_x"],
            tile_extent_y=slide_metadata["tile_extent_y"],
            tiles=region_tiles,
            x=torch.from_numpy(region_tiles["x"].to_numpy()),
            y=torch.from_numpy(region_tiles["y"].to_numpy()),
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
        folder_embeddings: str | None = None,
    ) -> None:
        super().__init__(
            uri=uri,
            uri_embeddings=uri_embeddings,
            mode=mode,
            minimum_region_size=minimum_region_size,
            folder_embeddings=folder_embeddings,
            include_labels=True,
        )


class EmbeddingsPredict(_Embeddings[MILPredictSample]):
    def __init__(
        self,
        uri: str,
        uri_embeddings: str,
        mode: EmbeddingsMode | str,
        minimum_region_size: int = 100,
        folder_embeddings: str | None = None,
    ) -> None:
        super().__init__(
            uri=uri,
            uri_embeddings=uri_embeddings,
            mode=mode,
            minimum_region_size=minimum_region_size,
            folder_embeddings=folder_embeddings,
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


def process_slides(slides: pd.DataFrame, mode: EmbeddingsMode) -> pd.DataFrame:
    match mode:
        case EmbeddingsMode.NEUTROPHILS:
            slides["neutrophils"] = slides["nancy_index"] >= 2
        case EmbeddingsMode.NANCY_LOW:
            slides = slides[slides["nancy_index"] < 2]
        case EmbeddingsMode.NANCY_HIGH:
            slides = slides[slides["nancy_index"] >= 2]
            slides["ulceration"] = slides["nancy_index"] == 4
            slides["nancy_index"] -= 2

    slides["name"] = slides["path"].apply(lambda x: Path(x).stem)
    return slides


def get_label(slide_metadata: pd.Series, mode: EmbeddingsMode) -> torch.Tensor:
    match mode:
        case EmbeddingsMode.NEUTROPHILS:
            return torch.tensor(slide_metadata["neutrophils"]).float()
        case EmbeddingsMode.NANCY_LOW:
            return torch.tensor(slide_metadata["nancy_index"]).float()
        case EmbeddingsMode.NANCY_HIGH:
            return torch.tensor(slide_metadata["ulceration"]).float()
