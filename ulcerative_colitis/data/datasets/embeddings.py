from collections.abc import Iterable, Sequence
from enum import Enum
from pathlib import Path
from typing import Generic, TypeVar

import h5py
import mlflow
import mlflow.artifacts
import pandas as pd
import torch
from rationai.mlkit.data.datasets import MetaTiledSlides
from torch.utils.data import ConcatDataset, Dataset

from ulcerative_colitis.typing import MetadataMIL, MILPredictSample, MILSample


T = TypeVar("T", bound=MILSample | MILPredictSample)


class EmbeddingsMode(Enum):
    NEUTROPHILS = "neutrophils"
    NANCY_HIGH = "nancy_high"
    NANCY_LOW = "nancy_low"


class Embeddings(MetaTiledSlides[MILSample]):
    def __init__(
        self,
        uri: str,
        uri_embeddings: str,
        mode: EmbeddingsMode | str,
        minimum_region_size: int = 100,
        folder_embeddings: str | None = None,
    ) -> None:
        if folder_embeddings is not None:
            self.folder_embeddings = Path(folder_embeddings)
        if not self.folder_embeddings.exists():
            self.folder_embeddings = Path(
                mlflow.artifacts.download_artifacts(uri_embeddings)
            )

        self.mode = EmbeddingsMode(mode)
        self.minimum_region_size = minimum_region_size
        super().__init__(uris=[uri])

    def generate_datasets(self) -> Iterable[Dataset[MILSample]]:
        self.slides = process_slides(self.slides, self.mode)
        return [
            EmbeddingsSlideBags(
                slide_metadata=slide,
                tiles=self.filter_tiles_by_slide(slide["id"]),
                file_embeddings=(self.folder_embeddings / slide["name"]).with_suffix(
                    ".h5"
                ),
                mode=self.mode,
                minimum_region_size=self.minimum_region_size,
                include_labels=True,
            )
            for _, slide in self.slides.iterrows()
        ]


class EmbeddingsPredict(MetaTiledSlides[MILPredictSample]):
    def __init__(
        self,
        uri: str,
        uri_embeddings: str,
        mode: EmbeddingsMode | str,
        minimum_region_size: int = 100,
        folder_embeddings: str | None = None,
    ) -> None:
        if folder_embeddings is not None:
            self.folder_embeddings = Path(folder_embeddings)
        if not self.folder_embeddings.exists():
            self.folder_embeddings = Path(
                mlflow.artifacts.download_artifacts(uri_embeddings)
            )

        self.mode = EmbeddingsMode(mode)
        self.minimum_region_size = minimum_region_size
        super().__init__(uris=[uri])

    def generate_datasets(self) -> Iterable[Dataset[MILSample]]:
        self.slides = process_slides(self.slides, self.mode)
        return [
            EmbeddingsSlideBags(
                slide_metadata=slide,
                tiles=self.filter_tiles_by_slide(slide["id"]),
                file_embeddings=(self.folder_embeddings / slide["name"]).with_suffix(
                    ".h5"
                ),
                mode=self.mode,
                minimum_region_size=self.minimum_region_size,
                include_labels=False,
            )
            for _, slide in self.slides.iterrows()
        ]


class EmbeddingsSlideBags(Dataset[T], Generic[T]):
    def __init__(
        self,
        slide_metadata: pd.Series,
        tiles: pd.DataFrame,
        file_embeddings: Path,
        mode: EmbeddingsMode | str,
        minimum_region_size: int = 100,
        include_labels: bool = True,
    ) -> None:
        self.slide_metadata = slide_metadata
        self.tiles = tiles.reset_index(drop=True)
        self.file_embeddings = file_embeddings
        self.mode = EmbeddingsMode(mode)
        self.include_labels = include_labels
        self.bags = self._create_bags(minimum_region_size)

    def _create_bags(self, minimum_region_size: int) -> pd.DataFrame:
        bags = self.tiles.groupby("region").agg("size")
        return bags[bags >= minimum_region_size].reset_index()

    def __len__(self) -> int:
        return len(self.bags)

    def __getitem__(self, idx: int) -> MILSample | MILPredictSample:
        region = self.bags.iloc[idx]["region"]
        region_tiles = self.tiles.query(f"region == {region}")

        with h5py.File(self.file_embeddings, "r") as f:
            dataset = f["embeddings"]
            assert isinstance(dataset, h5py.Dataset)
            region_embeddings = torch.from_numpy(dataset[region_tiles.index.to_numpy()])

        metadata = MetadataMIL(
            slide=self.slide_metadata["name"],
            slide_path=Path(self.slide_metadata["path"]),
            level=self.slide_metadata["level"],
            tile_extent_x=self.slide_metadata["tile_extent_x"],
            tile_extent_y=self.slide_metadata["tile_extent_y"],
            tiles=region_tiles,
            x=torch.from_numpy(region_tiles["x"].to_numpy()),
            y=torch.from_numpy(region_tiles["y"].to_numpy()),
        )

        if not self.include_labels:
            return region_embeddings, metadata

        label = get_label(self.slide_metadata, self.mode)

        return region_embeddings, label, metadata


class EmbeddingsSubset(ConcatDataset):
    def __init__(self, datasets: list[Dataset], slides: pd.DataFrame) -> None:
        super().__init__(datasets)
        self.slides = slides


def create_embeddings_subset(
    dataset: Embeddings,
    slide_indices: Sequence[int],
) -> EmbeddingsSubset:
    selected_datasets = [dataset.datasets[i] for i in slide_indices]
    subset_slides = dataset.slides.iloc[list(slide_indices)].reset_index(drop=True)
    return EmbeddingsSubset(selected_datasets, subset_slides)


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
