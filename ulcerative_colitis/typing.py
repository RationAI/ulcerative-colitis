from typing import TypeAlias, TypedDict

from torch import Tensor


class Metadata(TypedDict):
    slide: str
    x: int
    y: int


Sample: TypeAlias = tuple[Tensor, Tensor, Metadata]
Input: TypeAlias = Sample
Outputs: TypeAlias = Tensor
