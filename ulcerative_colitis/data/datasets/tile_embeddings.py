import warnings
from collections.abc import Sequence
from pathlib import Path
from typing import Generic, TypeVar, cast

import mlflow
import mlflow.artifacts
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, Subset

from ulcerative_colitis.data.datasets.tiles import LabelMode, get_label, process_slides
from ulcerative_colitis.typing import MetadataMIL, MILPredictSample, MILSample


T = TypeVar("T", bound=MILSample | MILPredictSample)


class _TileEmbeddings(Dataset[T], Generic[T]):
    def __init__(
        self,
        tiling_uri: str,
        embeddings_uri: str,
        mode: LabelMode | str | None = None,
        embeddings_folder: Path | str | None = None,
        padding: bool = True,
    ) -> None:
        self.mode = LabelMode(mode) if mode is not None else None
        artifacts = Path(mlflow.artifacts.download_artifacts(tiling_uri))
        self.tiles = pd.read_parquet(artifacts / "tiles.parquet")
        self.slides = pd.read_parquet(artifacts / "slides.parquet")
        self.slides = process_slides(self.slides, self.mode)

        if embeddings_folder is None or not Path(embeddings_folder).exists():
            self.embeddings_folder = Path(
                mlflow.artifacts.download_artifacts(embeddings_uri)
            )
        else:
            self.embeddings_folder = Path(embeddings_folder)

        self.padding = padding
        self.max_embeddings = self.tiles["slide_id"].value_counts().max()

    def __len__(self) -> int:
        return len(self.slides)

    def __getitem__(self, idx: int) -> T:
        slide_metadata = self.slides.iloc[idx]
        tiles = self.tiles[self.tiles["slide_id"] == slide_metadata["id"]]
        slide_name = str(slide_metadata["name"])
        embeddings_dict = cast(
            "dict[str, torch.Tensor]",
            torch.load(
                (self.embeddings_folder / slide_name).with_suffix(".pt"),
                map_location="cpu",
            ),
        )

        embeddings = align_tile_embeddings(tiles, embeddings_dict)
        pad_amount = self.max_embeddings - embeddings.shape[0]
        if self.padding:
            embeddings = F.pad(embeddings, (0, 0, 0, pad_amount), value=0.0)

        metadata = MetadataMIL(
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

        if self.mode is None:
            return embeddings, metadata  # type: ignore[return-value]

        label = get_label(slide_metadata, self.mode)

        return embeddings, label, metadata  # type: ignore[return-value]


class TileEmbeddings(_TileEmbeddings[MILSample]):
    def __init__(
        self,
        tiling_uri: str,
        embeddings_uri: str,
        mode: LabelMode | str,
        embeddings_folder: Path | str | None = None,
        padding: bool = True,
    ) -> None:
        super().__init__(
            tiling_uri=tiling_uri,
            embeddings_uri=embeddings_uri,
            embeddings_folder=embeddings_folder,
            mode=mode,
            padding=padding,
        )


class TileEmbeddingsPredict(_TileEmbeddings[MILPredictSample]):
    def __init__(
        self,
        tiling_uri: str,
        embeddings_uri: str,
        mode: LabelMode | str | None = None,
        embeddings_folder: Path | str | None = None,
        padding: bool = True,
    ) -> None:
        super().__init__(
            tiling_uri=tiling_uri,
            embeddings_uri=embeddings_uri,
            embeddings_folder=embeddings_folder,
            mode=mode,
            padding=padding,
        )


class TileEmbeddingsSubset(Subset[MILSample]):
    def __init__(
        self,
        dataset: TileEmbeddings,
        indices: Sequence[int],
    ) -> None:
        super().__init__(dataset, indices)
        self.slides = dataset.slides.iloc[list(indices)].reset_index()
        self.tiles = dataset.tiles.query(f"slide_id in {tuple(self.slides['id'])}")


def align_tile_embeddings(
    tiles: pd.DataFrame, embeddings_dict: dict[str, torch.Tensor]
) -> torch.Tensor:
    if (tiles["x"] == embeddings_dict["x_coords"].numpy()).all() and (
        tiles["y"] == embeddings_dict["y_coords"].numpy()
    ).all():
        return embeddings_dict["embeddings"]

    warnings.warn(
        "Tile coordinates are not aligned with embeddings coordinates.", stacklevel=2
    )

    embeddings_df = pd.DataFrame(
        {
            "x": embeddings_dict["x_coords"].numpy(),
            "y": embeddings_dict["y_coords"].numpy(),
            "embeddings": list(embeddings_dict["embeddings"].numpy()),
        }
    )

    merged = tiles.merge(embeddings_df, on=["x", "y"], how="left")

    if merged["embeddings"].isnull().any():
        raise ValueError("Some tiles do not have corresponding embeddings.")

    return torch.from_numpy(np.stack(merged["embeddings"].tolist()))
