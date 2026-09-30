from pathlib import Path
from typing import TypeAlias, TypedDict

from datasets import Dataset as HFDataset
from torch import Tensor


class Metadata(TypedDict):
    slide_name: str


class MetadataBags(Metadata):
    slide_path: Path
    level: int
    tile_extent_x: int
    tile_extent_y: int
    tiles: HFDataset
    x: Tensor  # Tensor[int]
    y: Tensor  # Tensor[int]


BagsSample: TypeAlias = tuple[Tensor, Tensor, MetadataBags]

BagsInput: TypeAlias = tuple[Tensor, Tensor, list[MetadataBags]]

Output: TypeAlias = Tensor
