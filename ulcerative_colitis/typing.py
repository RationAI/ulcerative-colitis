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


TrainSample: TypeAlias = tuple[Tensor, Tensor, TrainMetadata]
TestSample: TypeAlias = tuple[Tensor, Tensor, Metadata]
PredictSample: TypeAlias = tuple[Tensor, Metadata]

TrainInput: TypeAlias = tuple[Tensor, Tensor, TrainMetadataBatch]
TestInput: TypeAlias = tuple[Tensor, Tensor, MetadataBatch]
PredictInput: TypeAlias = tuple[Tensor, MetadataBatch]

Output: TypeAlias = Tensor
