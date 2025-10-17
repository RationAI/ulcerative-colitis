from pathlib import Path
from typing import TypeAlias, TypedDict

import pandas as pd
from torch import Tensor


class Metadata(TypedDict):
    slide_id: str


class MetadataTiles(Metadata):
    slide_id: str
    x: int
    y: int


TilesSample: TypeAlias = tuple[Tensor, Tensor, MetadataTiles]
TilesPredictSample: TypeAlias = tuple[Tensor, MetadataTiles]


class MetadataTileEmbeddings(Metadata):
    slide_id: str
    slide_name: str
    slide_path: Path
    level: int
    tile_extent_x: int
    tile_extent_y: int
    tiles: pd.DataFrame
    x: Tensor  # Tensor[int]
    y: Tensor  # Tensor[int]


TileEmbeddingsSample: TypeAlias = tuple[Tensor, Tensor, MetadataTileEmbeddings]
TileEmbeddingsPredictSample: TypeAlias = tuple[Tensor, MetadataTileEmbeddings]

TileEmbeddingsInput: TypeAlias = tuple[Tensor, Tensor, list[MetadataTileEmbeddings]]
TileEmbeddingsPredictInput: TypeAlias = tuple[Tensor, list[MetadataTileEmbeddings]]


class MetadataSlideEmbeddings(Metadata):
    slide_id: str
    slide_name: str
    slide_path: Path


SlideEmbeddingsSample: TypeAlias = tuple[Tensor, Tensor, MetadataSlideEmbeddings]
SlideEmbeddingsPredictSample: TypeAlias = tuple[Tensor, MetadataSlideEmbeddings]

SlideEmbeddingsInput: TypeAlias = tuple[Tensor, Tensor, list[MetadataSlideEmbeddings]]
SlideEmbeddingsPredictInput: TypeAlias = tuple[Tensor, list[MetadataSlideEmbeddings]]

Output: TypeAlias = Tensor
