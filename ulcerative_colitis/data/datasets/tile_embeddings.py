import warnings
from collections.abc import Iterable
from itertools import repeat
from pathlib import Path
from typing import Generic, TypeVar

import mlflow
import mlflow.artifacts
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from ulcerative_colitis.data.datasets.labels import LabelMode, get_label, process_slides
from ulcerative_colitis.typing import (
    MetadataTileEmbeddings,
    TileEmbeddingsPredictSample,
    TileEmbeddingsSample,
)


T = TypeVar("T", bound=TileEmbeddingsSample | TileEmbeddingsPredictSample)


class _TileEmbeddings(Dataset[T], Generic[T]):
    def __init__(
        self,
        tiling_uris: Iterable[str],
        embeddings_uris: Iterable[str],
        mode: LabelMode | str | None = None,
        embeddings_folders: Iterable[Path | str | None] | None = None,
        padding: bool = True,
        stride_eq_tile: bool = False,
        include_labels: bool = True,
    ) -> None:
        self.mode = LabelMode(mode) if mode is not None else None
        self.include_labels = include_labels
        self.stride_eq_tile = stride_eq_tile

        if self.include_labels and self.mode is None:
            raise ValueError("Mode must be specified when including labels.")

        self.slides, self.tiles = self.download_artifacts(
            tiling_uris, embeddings_uris, embeddings_folders
        )
        self.tiles = self.filter_tiles_by_stride()
        self.slides = self.filter_empty_slides()
        self.slides = process_slides(self.slides, self.mode)

        self.padding = padding
        self.max_embeddings = self.tiles["slide_id"].value_counts().max()

    def download_artifacts(
        self,
        tiling_uris: Iterable[str],
        embeddings_uris: Iterable[str],
        embeddings_folders: Iterable[Path | str | None] | None,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        if embeddings_folders is None:
            embeddings_folders = repeat(None)

        slide_dfs = []
        tile_dfs = []
        for tiling_uri, embeddings_uri, embeddings_folder in zip(
            tiling_uris, embeddings_uris, embeddings_folders, strict=False
        ):
            slide_dfs.append(
                pd.read_parquet(
                    Path(mlflow.artifacts.download_artifacts(tiling_uri))
                    / "slides.parquet"
                )
            )
            tile_dfs.append(
                pd.read_parquet(
                    Path(mlflow.artifacts.download_artifacts(tiling_uri))
                    / "tiles.parquet"
                )
            )

            if embeddings_folder is None:
                embeddings_folder = Path(
                    mlflow.artifacts.download_artifacts(embeddings_uri)
                )
            embeddings_folder = Path(embeddings_folder)

            slide_dfs[-1]["embeddings_folder"] = embeddings_folder

        return (
            pd.concat(slide_dfs, ignore_index=True),
            pd.concat(tile_dfs, ignore_index=True),
        )

    def filter_empty_slides(self) -> pd.DataFrame:
        valid_slide_ids = self.tiles["slide_id"].unique()
        filtered_slides = self.slides[
            self.slides["id"].isin(valid_slide_ids)
        ].reset_index(drop=True)
        return filtered_slides

    def filter_tiles_by_stride(self) -> pd.DataFrame:
        if not self.stride_eq_tile:
            return self.tiles

        filtered_tiles = self.tiles[
            (self.tiles["x"] % 224 == 0) & (self.tiles["y"] % 224 == 0)
        ].reset_index(drop=True)
        return filtered_tiles

    def __len__(self) -> int:
        return len(self.slides)

    def __getitem__(self, idx: int) -> T:
        slide_metadata = self.slides.iloc[idx]
        tiles = self.tiles[self.tiles["slide_id"] == slide_metadata["id"]]
        slide_name = str(slide_metadata["name"])
        embeddings_df = pd.read_parquet(
            slide_metadata["embeddings_folder"] / f"{slide_name}.parquet"
        )

        embeddings = align_tile_embeddings(tiles, embeddings_df, self.stride_eq_tile)
        pad_amount = self.max_embeddings - embeddings.shape[0]
        if self.padding:
            embeddings = F.pad(embeddings, (0, 0, 0, pad_amount), value=0.0)

        metadata = MetadataTileEmbeddings(
            slide_id=slide_metadata["id"],
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
            return embeddings, metadata  # type: ignore[return-value]

        assert self.mode is not None
        label = get_label(slide_metadata, self.mode)

        return embeddings, label, metadata  # type: ignore[return-value]


class TileEmbeddings(_TileEmbeddings[TileEmbeddingsSample]):
    def __init__(
        self,
        tiling_uris: Iterable[str],
        embeddings_uris: Iterable[str],
        mode: LabelMode | str,
        embeddings_folders: Iterable[Path | str | None] | None = None,
        padding: bool = True,
        stride_eq_tile: bool = False,
    ) -> None:
        super().__init__(
            tiling_uris=tiling_uris,
            embeddings_uris=embeddings_uris,
            embeddings_folders=embeddings_folders,
            mode=mode,
            padding=padding,
            stride_eq_tile=stride_eq_tile,
            include_labels=True,
        )


class TileEmbeddingsPredict(_TileEmbeddings[TileEmbeddingsPredictSample]):
    def __init__(
        self,
        tiling_uris: Iterable[str],
        embeddings_uris: Iterable[str],
        mode: LabelMode | str | None = None,
        embeddings_folders: Iterable[Path | str | None] | None = None,
        padding: bool = True,
        stride_eq_tile: bool = False,
    ) -> None:
        super().__init__(
            tiling_uris=tiling_uris,
            embeddings_uris=embeddings_uris,
            embeddings_folders=embeddings_folders,
            mode=mode,
            padding=padding,
            stride_eq_tile=stride_eq_tile,
            include_labels=False,
        )


def align_tile_embeddings(
    tiles: pd.DataFrame, embeddings_df: pd.DataFrame, stride_eq_tile: bool
) -> torch.Tensor:
    if stride_eq_tile:
        embeddings_df = embeddings_df[
            (embeddings_df["x"] % 224 == 0) & (embeddings_df["y"] % 224 == 0)
        ].reset_index(drop=True)

    print(tiles)
    print(embeddings_df)

    print(tiles["x"])
    print(embeddings_df["x"])
    if (tiles["x"] == embeddings_df["x"]).all() and (
        tiles["y"] == embeddings_df["y"]
    ).all():
        return torch.from_numpy(np.stack(embeddings_df["embeddings"].tolist()))

    warnings.warn(
        "Tile coordinates are not aligned with embeddings coordinates.", stacklevel=2
    )

    merged = tiles.merge(embeddings_df, on=["x", "y"], how="left")

    if merged["embeddings"].isnull().any():
        raise ValueError("Some tiles do not have corresponding embeddings.")

    return torch.from_numpy(np.stack(merged["embeddings"].tolist()))
