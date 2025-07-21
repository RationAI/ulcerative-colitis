from typing import TypeAlias, TypedDict

import pandas as pd
from torch import Tensor


class Metadata(TypedDict):
    slide: str
    x: int
    y: int


class MetadataMIL(TypedDict):
    slide: str
    slide_path: str
    level: int
    tile_extent_x: int
    tile_extent_y: int
    tiles: pd.DataFrame
    x: Tensor  # Tensor[int]
    y: Tensor  # Tensor[int]


class MetadataMILBatch(TypedDict):
    slide: list[str]
    slide_path: list[str]
    level: Tensor  # Tensor[int]
    tile_extent_x: Tensor  # Tensor[int]
    tile_extent_y: Tensor  # Tensor[int]
    tiles: pd.DataFrame
    x: Tensor  # Tensor[int]
    y: Tensor  # Tensor[int]


class MetadataBatch(TypedDict):
    slide: list[str]
    x: Tensor  # Tensor[int]
    y: Tensor  # Tensor[int]


class TrainMetadata(TypedDict):
    slide: list[str]
    x: Tensor  # Tensor[int]
    y: Tensor  # Tensor[int]


class TrainMetadataBatch(TypedDict):
    slide: list[list[str]]
    x: Tensor  # Tensor[int]
    y: Tensor  # Tensor[int]


TrainSample: TypeAlias = tuple[Tensor, Tensor, TrainMetadata]
TestSample: TypeAlias = tuple[Tensor, Tensor, Metadata]
PredictSample: TypeAlias = tuple[Tensor, Metadata]

MILSample: TypeAlias = tuple[Tensor, Tensor, MetadataMIL]
MILPredictSample: TypeAlias = tuple[Tensor, MetadataMIL]

TrainInput: TypeAlias = tuple[Tensor, Tensor, TrainMetadataBatch]
TestInput: TypeAlias = tuple[Tensor, Tensor, MetadataBatch]
PredictInput: TypeAlias = tuple[Tensor, MetadataBatch]

MILInput: TypeAlias = tuple[Tensor, Tensor, MetadataMILBatch]
MILPredictInput: TypeAlias = tuple[Tensor, MetadataMILBatch]

Output: TypeAlias = Tensor
