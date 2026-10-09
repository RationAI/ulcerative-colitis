from pathlib import Path
from typing import TypeAlias, TypedDict

from datasets import Dataset as HFDataset
from torch import Tensor


class MetadataBag(TypedDict):
    slide_name: str
    slide_path: Path
    level: int
    tile_extent_x: int
    tile_extent_y: int
    tiles: HFDataset
    x: Tensor  # Tensor[int]
    y: Tensor  # Tensor[int]


BagsSample: TypeAlias = tuple[Tensor, Tensor, MetadataBag]
BagsPredictSample: TypeAlias = tuple[Tensor, MetadataBag]

BagsInput: TypeAlias = tuple[Tensor, Tensor, list[MetadataBag]]
BagsPredictInput: TypeAlias = tuple[Tensor, list[MetadataBag]]

Output: TypeAlias = Tensor
