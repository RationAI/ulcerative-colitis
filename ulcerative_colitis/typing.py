from typing import TypeAlias, TypedDict

from torch import Tensor


class Metadata(TypedDict):
    slide: str
    x: int
    y: int


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


Sample: TypeAlias = tuple[Tensor, Tensor, Metadata]
TrainSample: TypeAlias = tuple[Tensor, Tensor, TrainMetadata]
PredictSample: TypeAlias = tuple[Tensor, Metadata]

Input: TypeAlias = tuple[Tensor, Tensor, MetadataBatch]
TrainInput: TypeAlias = tuple[Tensor, Tensor, TrainMetadataBatch]
PredictInput: TypeAlias = tuple[Tensor, MetadataBatch]

Output: TypeAlias = Tensor
