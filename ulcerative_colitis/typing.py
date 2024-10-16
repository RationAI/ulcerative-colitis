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


Sample: TypeAlias = tuple[Tensor, Tensor, Metadata]
PredictSample: TypeAlias = tuple[Tensor, Metadata]

Input: TypeAlias = tuple[Tensor, Tensor, MetadataBatch]
PredictInput: TypeAlias = tuple[Tensor, MetadataBatch]

Output: TypeAlias = Tensor
