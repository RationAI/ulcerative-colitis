from pathlib import Path
from typing import Generic, TypeVar

import mlflow.artifacts
import pandas as pd
import torch
from torch.utils.data import Dataset

from ulcerative_colitis.data.datasets.labels import LabelMode, get_label, process_slides
from ulcerative_colitis.typing import (
    MetadataSlideEmbeddings,
    SlideEmbeddingsPredictSample,
    SlideEmbeddingsSample,
)


T = TypeVar("T", bound=SlideEmbeddingsSample | SlideEmbeddingsPredictSample)


class _SlideEmbeddings(Dataset[T], Generic[T]):
    def __init__(
        self,
        tiling_uri: str,
        slide_embeddings_uri: str,
        mode: LabelMode | str | None = None,
        include_labels: bool = True,
    ) -> None:
        self.mode = LabelMode(mode) if mode is not None else None
        self.include_labels = include_labels

        if self.include_labels and self.mode is None:
            raise ValueError("Mode must be specified when including labels.")

        artifacts_tiling = Path(mlflow.artifacts.download_artifacts(tiling_uri))
        artifacts_slide_embeddings = Path(
            mlflow.artifacts.download_artifacts(slide_embeddings_uri)
        )
        self.slides = pd.read_parquet(artifacts_tiling / "slides.parquet")
        self.slide_embeddings = pd.read_parquet(
            artifacts_slide_embeddings / "slide_embeddings.parquet",
        ).set_index("slide_id")

        self.slides = process_slides(self.slides, self.mode)

    def __len__(self) -> int:
        return len(self.slides)

    def __getitem__(self, idx: int) -> T:
        slide_metadata = self.slides.iloc[idx]
        slide_embedding = torch.from_numpy(
            self.slide_embeddings.loc[slide_metadata["id"]]["embedding"]
        ).float()

        metadata = MetadataSlideEmbeddings(
            slide_id=slide_metadata["id"],
            slide_name=slide_metadata["name"],
            slide_path=Path(slide_metadata["path"]),
        )

        if self.mode is None:
            return slide_embedding, metadata  # type: ignore[return-value]

        label = get_label(slide_metadata, self.mode)
        return slide_embedding, label, metadata  # type: ignore[return-value]


class SlideEmbeddings(_SlideEmbeddings[SlideEmbeddingsSample]):
    def __init__(
        self,
        tiling_uri: str,
        slide_embeddings_uri: str,
        mode: LabelMode | str,
    ) -> None:
        super().__init__(
            tiling_uri=tiling_uri,
            slide_embeddings_uri=slide_embeddings_uri,
            mode=mode,
            include_labels=True,
        )


class SlideEmbeddingsPredict(_SlideEmbeddings[SlideEmbeddingsPredictSample]):
    def __init__(
        self,
        tiling_uri: str,
        slide_embeddings_uri: str,
        mode: LabelMode | str | None = None,
    ) -> None:
        super().__init__(
            tiling_uri=tiling_uri,
            slide_embeddings_uri=slide_embeddings_uri,
            mode=mode,
            include_labels=False,
        )
