from pathlib import Path
from typing import TypeAlias, TypedDict

from datasets import Dataset as HFDataset
from torch import Tensor


class Metadata(TypedDict):
    slide_name: str


class MetadataBatch(TypedDict):
    slide_name: list[str]


class MetadataTiles(Metadata):
    x: int
    y: int


class MetadataTilesBatch(MetadataBatch):
    x: Tensor
    y: Tensor


TilesSample: TypeAlias = tuple[Tensor, Tensor, MetadataTiles]
TilesPredictSample: TypeAlias = tuple[Tensor, MetadataTiles]

TilesInput: TypeAlias = tuple[Tensor, Tensor, MetadataTilesBatch]
TilesPredictInput: TypeAlias = tuple[Tensor, MetadataTilesBatch]

MetadataEmbeddings: TypeAlias = MetadataTiles
MetadataEmbeddingsBatch: TypeAlias = MetadataTilesBatch

EmbeddingsSample: TypeAlias = tuple[Tensor, Tensor, MetadataEmbeddings]
EmbeddingsPredictSample: TypeAlias = tuple[Tensor, MetadataEmbeddings]

EmbeddingsInput: TypeAlias = tuple[Tensor, Tensor, MetadataEmbeddingsBatch]
EmbeddingsPredictInput: TypeAlias = tuple[Tensor, MetadataEmbeddingsBatch]


class MetadataBags(Metadata):
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
