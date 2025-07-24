from pathlib import Path
from typing import TypeAlias, TypedDict

import pandas as pd
from torch import Tensor


class Metadata(TypedDict):
    slide: str
    x: int
    y: int


Sample: TypeAlias = tuple[Tensor, Tensor, Metadata]
PredictSample: TypeAlias = tuple[Tensor, Metadata]


class MetadataMIL(TypedDict):
    slide: str
    slide_path: Path
    level: int
    tile_extent_x: int
    tile_extent_y: int
    tiles: pd.DataFrame
    x: Tensor  # Tensor[int]
    y: Tensor  # Tensor[int]


MILSample: TypeAlias = tuple[Tensor, Tensor, MetadataMIL]
MILPredictSample: TypeAlias = tuple[Tensor, MetadataMIL]

MILInput: TypeAlias = tuple[Tensor, Tensor, list[MetadataMIL]]
MILPredictInput: TypeAlias = tuple[Tensor, list[MetadataMIL]]

Output: TypeAlias = Tensor
