from collections.abc import Iterable
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
        tiling_uris: Iterable[str],
        slide_embeddings_uris: Iterable[str],
        mode: LabelMode | str | None = None,
        include_labels: bool = True,
    ) -> None:
        self.mode = LabelMode(mode) if mode is not None else None
        self.include_labels = include_labels

        if self.include_labels and self.mode is None:
            raise ValueError("Mode must be specified when including labels.")

        slides, slide_embeddings = self.download_artifacts(
            tiling_uris, slide_embeddings_uris
        )

        self.slides = slides.merge(
            slide_embeddings, how="left", left_on="id", right_index=True
        )

        self.slides = process_slides(self.slides, self.mode)

    def download_artifacts(
        self, tiling_uris: Iterable[str], slide_embeddings_uris: Iterable[str]
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        slide_dfs = []
        for tiling_uri in tiling_uris:
            slide_dfs.append(
                pd.read_parquet(
                    Path(mlflow.artifacts.download_artifacts(tiling_uri))
                    / "slides.parquet"
                )
            )

        slide_embeddings_dfs = []
        for slide_embeddings_uri in slide_embeddings_uris:
            slide_embeddings_dfs.append(
                pd.read_parquet(
                    Path(mlflow.artifacts.download_artifacts(slide_embeddings_uri))
                    / "slide_embeddings.parquet"
                ).set_index("slide_id")
            )

        return pd.concat(slide_dfs, ignore_index=True), pd.concat(slide_embeddings_dfs)

    def __len__(self) -> int:
        return len(self.slides)

    def __getitem__(self, idx: int) -> T:
        slide_metadata = self.slides.iloc[idx]
        slide_embedding = torch.from_numpy(slide_metadata["embedding"]).float()

        metadata = MetadataSlideEmbeddings(
            slide_id=slide_metadata["id"],
            slide_name=slide_metadata["name"],
            slide_path=Path(slide_metadata["path"]),
        )

        if self.mode is None:
            return slide_embedding, metadata  # type: ignore[return-value]

        label = get_label(slide_metadata, self.mode).unsqueeze(-1)
        return slide_embedding, label, metadata  # type: ignore[return-value]


class SlideEmbeddings(_SlideEmbeddings[SlideEmbeddingsSample]):
    def __init__(
        self,
        tiling_uris: Iterable[str],
        slide_embeddings_uris: Iterable[str],
        mode: LabelMode | str,
    ) -> None:
        super().__init__(
            tiling_uris=tiling_uris,
            slide_embeddings_uris=slide_embeddings_uris,
            mode=mode,
            include_labels=True,
        )


class SlideEmbeddingsPredict(_SlideEmbeddings[SlideEmbeddingsPredictSample]):
    def __init__(
        self,
        tiling_uris: Iterable[str],
        slide_embeddings_uris: Iterable[str],
        mode: LabelMode | str | None = None,
    ) -> None:
        super().__init__(
            tiling_uris=tiling_uris,
            slide_embeddings_uris=slide_embeddings_uris,
            mode=mode,
            include_labels=False,
        )
