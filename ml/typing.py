from pathlib import Path
from typing import TypeAlias, TypedDict

from datasets import Dataset as HFDataset
from torch import Tensor


class Metadata(TypedDict):
    slide_id: str


class MetadataTiles(Metadata):
    x: int
    y: int


TilesSample: TypeAlias = tuple[Tensor, Tensor, MetadataTiles]
TilesPredictSample: TypeAlias = tuple[Tensor, MetadataTiles]


class MetadataEmbeddings(Metadata):
    x: int
    y: int


EmbeddingsSample: TypeAlias = tuple[Tensor, Tensor, MetadataEmbeddings]
EmbeddingsPredictSample: TypeAlias = tuple[Tensor, MetadataEmbeddings]


class MetadataBags(Metadata):
    slide_name: str
    slide_path: Path
    level: int
    tile_extent_x: int
    tile_extent_y: int
    tiles: HFDataset
    x: Tensor  # Tensor[int]
    y: Tensor  # Tensor[int]


BagsSample: TypeAlias = tuple[Tensor, Tensor, MetadataBags]
BagsPredictSample: TypeAlias = tuple[Tensor, MetadataBags]

BagsInput: TypeAlias = tuple[Tensor, Tensor, list[MetadataBags]]
BagsPredictInput: TypeAlias = tuple[Tensor, list[MetadataBags]]

Output: TypeAlias = Tensor
