from collections.abc import Iterable
from enum import Enum
from pathlib import Path
from typing import Generic, TypeVar, cast

import mlflow
import mlflow.artifacts
import pandas as pd
import torch
from rationai.mlkit.data.datasets import MetaTiledSlides
from torch.utils.data import Dataset

from ulcerative_colitis.typing import MetadataMIL, MILPredictSample, MILSample


T = TypeVar("T", bound=MILSample | MILPredictSample)

LOCATIONS = ("cekoascendens", "descendens", "rektosigma", "transverzum")


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
    ) -> None:
        self.folder = Path(mlflow.artifacts.download_artifacts(uri_embeddings))
        self.mode = EmbeddingsMode(mode)
        super().__init__(uris=[uri])

    def generate_datasets(self) -> Iterable[Dataset[MILSample]]:
        self.slides = process_slides(self.slides, self.mode)
        return [
            _Embeddings[MILSample](
                self.slides,
                self.tiles,
                self.folder,
                self.mode,
            )
        ]


class EmbeddingsPredict(MetaTiledSlides[MILPredictSample]):
    def __init__(
        self,
        uri: str,
        uri_embeddings: str,
        mode: EmbeddingsMode,
        slide_names: list[str] | None = None,
    ) -> None:
        self.folder = Path(mlflow.artifacts.download_artifacts(uri_embeddings))
        self.mode = mode
        self.slide_names = slide_names
        super().__init__(uris=[uri])

    def generate_datasets(self) -> Iterable[Dataset[MILPredictSample]]:
        self.slides = process_slides(self.slides, self.mode, self.slide_names)
        return [
            _Embeddings[MILPredictSample](
                self.slides,
                self.tiles,
                self.folder,
                self.mode,
                include_labels=False,
            )
        ]


class _Embeddings(Dataset[T], Generic[T]):
    def __init__(
        self,
        slides: pd.DataFrame,
        tiles_all: pd.DataFrame,
        folder: Path,
        mode: EmbeddingsMode,
        include_labels: bool = True,
    ) -> None:
        super().__init__()
        self.slides = slides
        self.tiles_all = tiles_all
        self.folder = folder
        self.mode = mode
        self.include_labels = include_labels

    def __len__(self) -> int:
        return len(self.slides)

    def __getitem__(self, idx: int) -> MILSample | MILPredictSample:
        slide_metadata = self.slides.iloc[idx]
        tiles = self.tiles_all.query(f"slide_id == {slide_metadata['id']!s}")
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


def get_slide_name(slide_metadata: pd.Series) -> str:
    return Path(slide_metadata["path"]).stem


def process_slides(
    slides: pd.DataFrame, mode: EmbeddingsMode, slide_names: list[str] | None = None
) -> pd.DataFrame:
    slides = slides[slides["location"].isin(LOCATIONS)].copy()

    match mode:
        case EmbeddingsMode.NEUTROPHILS:
            slides["neutrophils"] = slides["nancy_index"] >= 2
        case EmbeddingsMode.NANCY_LOW:
            slides = slides[slides["nancy_index"] < 2]
        case EmbeddingsMode.NANCY_HIGH:
            slides = slides[slides["nancy_index"] >= 2]
            slides["nancy_index"] -= 2

    if slide_names is not None:
        slides = slides[slides["path"].str.contains("|".join(slide_names))]
    return slides


def get_label(slide_metadata: pd.Series, mode: EmbeddingsMode) -> torch.Tensor:
    match mode:
        case EmbeddingsMode.NEUTROPHILS:
            return torch.tensor(slide_metadata["neutrophils"]).float()
        case EmbeddingsMode.NANCY_LOW | EmbeddingsMode.NANCY_HIGH:
            return torch.tensor(slide_metadata["nancy_index"]).float()
