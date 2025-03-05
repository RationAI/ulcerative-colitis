import random
from collections.abc import Iterable
from pathlib import Path
from typing import Generic, TypeVar

import mlflow
import mlflow.artifacts
import pandas as pd
import torch
from rationai.mlkit.data.datasets import MetaTiledSlides
from torch.utils.data import Dataset

from ulcerative_colitis.typing import (
    Metadata,
    PredictSample,
    TestSample,
    TrainMetadata,
    TrainSample,
)


T = TypeVar("T", bound=TestSample | PredictSample)
R = TypeVar("R", bound=TrainSample | PredictSample | TestSample)

LOCATIONS = ("cekoascendens", "descendens", "rektosigma", "transverzum")


class Embeddings(MetaTiledSlides[R]):
    def __init__(
        self,
        uri: str,
        uri_embeddings: str,
    ) -> None:
        self.folder = Path(mlflow.artifacts.download_artifacts(uri_embeddings))
        super().__init__(uris=[uri])


class EmbeddingsTrain(Embeddings[TrainSample]):
    def __init__(
        self,
        uri: str,
        uri_embeddings: str,
        inner_batch_size: int,
    ) -> None:
        self.inner_batch_size = inner_batch_size
        super().__init__(uri=uri, uri_embeddings=uri_embeddings)

    def generate_datasets(self) -> Iterable[Dataset[TrainSample]]:
        self.slides = process_slides(self.slides)
        return [
            _EmbeddingsSlideTilesTrain(
                self.slides,
                self.tiles,
                self.folder,
                self.inner_batch_size,
            )
        ]


class EmbeddingsTest(Embeddings[TestSample]):
    def generate_datasets(self) -> Iterable[Dataset[TestSample]]:
        self.slides = process_slides(self.slides)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return [
            _EmbeddingsSlideTiles(
                slide,
                torch.load(
                    (self.folder / get_slide_name(slide)).with_suffix(".pt"),
                    map_location=device,
                ),
                tiles=self.filter_tiles_by_slide(slide["id"]),
                include_labels=True,
            )
            for _, slide in self.slides.iterrows()
        ]


class EmbeddingsPredict(Embeddings[PredictSample]):
    def generate_datasets(self) -> Iterable[Dataset[TestSample]]:
        self.slides = process_slides(self.slides)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return [
            _EmbeddingsSlideTiles(
                slide,
                torch.load(
                    (self.folder / get_slide_name(slide)).with_suffix(".pt"),
                    map_location=device,
                ),
                tiles=self.filter_tiles_by_slide(slide["id"]),
                include_labels=False,
            )
            for _, slide in self.slides.iterrows()
        ]


class _EmbeddingsSlideTiles(Dataset[T], Generic[T]):
    def __init__(
        self,
        slide_metadata: pd.Series,
        embedings: torch.Tensor,
        tiles: pd.DataFrame,
        include_labels: bool = True,
    ) -> None:
        super().__init__()
        self.slide_metadata = slide_metadata
        self.embedings = embedings
        self.tiles = tiles
        self.include_labels = include_labels

    def __len__(self) -> int:
        return len(self.embedings)

    def __getitem__(self, idx: int) -> TestSample | PredictSample:
        vector = self.embedings[idx]
        tile = self.tiles.iloc[idx]
        metadata = Metadata(
            slide=get_slide_name(self.slide_metadata), x=tile["x"], y=tile["y"]
        )

        if not self.include_labels:
            return vector, metadata

        label = torch.tensor(self.slide_metadata["nancy_index"])
        return vector, label, metadata


class _EmbeddingsSlideTilesTrain(Dataset[TrainSample]):
    def __init__(
        self,
        slides: pd.DataFrame,
        tiles_all: pd.DataFrame,
        folder: Path,
        inner_batch_size: int,
    ) -> None:
        super().__init__()
        self.slides = slides
        self.tiles_all = tiles_all
        self.folder = folder
        self.inner_batch_size = inner_batch_size

    def __len__(self) -> int:
        return len(self.slides)

    def __getitem__(self, idx: int) -> TrainSample:
        slide_metadata = self.slides.iloc[idx]
        tiles = self.tiles_all.query(f"slide_id == {slide_metadata['id']!s}")
        slide_name = get_slide_name(slide_metadata)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        embeddings = torch.load(
            (self.folder / slide_name).with_suffix(".pt"), map_location=device
        )

        vectors = []
        xs = []
        ys = []
        for _ in range(self.inner_batch_size):
            i = random.randint(0, len(tiles) - 1)
            vectors.append(embeddings[i])

            xs.append(tiles.iloc[i]["x"])
            ys.append(tiles.iloc[i]["y"])

        label = torch.tensor(slide_metadata["nancy_index"])
        metadata = TrainMetadata(
            slide=[slide_name] * self.inner_batch_size,
            x=torch.tensor(xs),
            y=torch.tensor(ys),
        )

        return torch.stack(vectors), label, metadata


def get_slide_name(slide_metadata: pd.Series) -> str:
    return Path(slide_metadata["path"]).stem


def process_slides(slides: pd.DataFrame) -> pd.DataFrame:
    return slides[slides["location"].isin(LOCATIONS)]
