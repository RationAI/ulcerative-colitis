from collections.abc import Sequence
from pathlib import Path
from typing import Generic, TypeVar, cast

import mlflow
import mlflow.artifacts
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, Subset

from ulcerative_colitis.data.datasets.tiles import LabelMode, get_label, process_slides
from ulcerative_colitis.typing import MetadataMIL, MILPredictSample, MILSample


T = TypeVar("T", bound=MILSample | MILPredictSample)


class _Embeddings(Dataset[T], Generic[T]):
    def __init__(
        self,
        uri: str,
        uri_embeddings: str,
        mode: LabelMode | str,
        folder_embeddings: Path | str | None = None,
        include_labels: bool = True,
    ) -> None:
        self.mode = LabelMode(mode)
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

        self.include_labels = include_labels
        self.max_embeddings = self.tiles["slide_id"].value_counts().max()

    def __len__(self) -> int:
        return len(self.slides)

    def __getitem__(self, idx: int) -> MILSample | MILPredictSample:
        slide_metadata = self.slides.iloc[idx]
        tiles = self.tiles.query(f"slide_id == {slide_metadata['id']!s}")
        slide_name = str(slide_metadata["name"])
        embeddings = cast(
            "torch.Tensor",
            torch.load(
                (self.folder_embeddings / slide_name).with_suffix(".pt"),
                map_location="cpu",
            ),
        )
        pad_amount = self.max_embeddings - embeddings.shape[0]
        embeddings = F.pad(embeddings, (0, 0, 0, pad_amount), value=0.0)

        metadata = MetadataMIL(
            slide_name=slide_name,
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
        mode: LabelMode | str,
        folder_embeddings: str | None = None,
    ) -> None:
        super().__init__(
            uri=uri,
            uri_embeddings=uri_embeddings,
            mode=mode,
            folder_embeddings=folder_embeddings,
            include_labels=True,
        )


class EmbeddingsPredict(_Embeddings[MILPredictSample]):
    def __init__(
        self,
        uri: str,
        uri_embeddings: str,
        mode: LabelMode | str,
        folder_embeddings: str | None = None,
    ) -> None:
        super().__init__(
            uri=uri,
            uri_embeddings=uri_embeddings,
            mode=mode,
            folder_embeddings=folder_embeddings,
            include_labels=False,
        )


class EmbeddingsSubset(Subset[MILSample]):
    def __init__(
        self,
        dataset: Embeddings,
        indices: Sequence[int],
    ) -> None:
        super().__init__(dataset, indices)
        self.slides = dataset.slides.iloc[list(indices)].reset_index()
        self.tiles = dataset.tiles.query(f"slide_id in {tuple(self.slides['id'])}")
