from collections.abc import Iterable
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


class Embeddings(MetaTiledSlides[MILSample]):
    def __init__(
        self,
        uri: str,
        uri_embeddings: str,
    ) -> None:
        self.folder = Path(mlflow.artifacts.download_artifacts(uri_embeddings))
        super().__init__(uris=[uri])

    def generate_datasets(self) -> Iterable[Dataset[MILSample]]:
        self.slides = process_slides(self.slides)
        return [
            _Embeddings[MILSample](
                self.slides,
                self.tiles,
                self.folder,
            )
        ]


class EmbeddingsPredict(MetaTiledSlides[MILPredictSample]):
    def __init__(
        self,
        uri: str,
        uri_embeddings: str,
    ) -> None:
        self.folder = Path(mlflow.artifacts.download_artifacts(uri_embeddings))
        super().__init__(uris=[uri])

    def generate_datasets(self) -> Iterable[Dataset[MILPredictSample]]:
        self.slides = process_slides(self.slides)
        return [
            _Embeddings[MILPredictSample](
                self.slides,
                self.tiles,
                self.folder,
                include_labels=False,
            )
        ]


class _Embeddings(Dataset[T], Generic[T]):
    def __init__(
        self,
        slides: pd.DataFrame,
        tiles_all: pd.DataFrame,
        folder: Path,
        include_labels: bool = True,
    ) -> None:
        super().__init__()
        self.slides = slides
        self.tiles_all = tiles_all
        self.folder = folder
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

        label = torch.tensor(slide_metadata["neutrophils"]).float()

        return embeddings, label, metadata


def get_slide_name(slide_metadata: pd.Series) -> str:
    return Path(slide_metadata["path"]).stem


def process_slides(slides: pd.DataFrame) -> pd.DataFrame:
    slides = slides[slides["location"].isin(LOCATIONS)].copy()
    slides["neutrophils"] = slides["nancy_index"] >= 2
    return slides
