from collections.abc import Sequence

from torch.utils.data import Subset

from ulcerative_colitis.data.datasets.slide_embeddings import SlideEmbeddings
from ulcerative_colitis.data.datasets.tile_embeddings import TileEmbeddings
from ulcerative_colitis.data.datasets.tiles import Tiles
from ulcerative_colitis.typing import (
    SlideEmbeddingsSample,
    TileEmbeddingsSample,
    TilesSample,
)


class TilesSubset(Subset[TilesSample]):
    def __init__(
        self,
        dataset: Tiles,
        indices: Sequence[int],
    ) -> None:
        super().__init__(dataset, indices)
        self.slides = dataset.slides.iloc[list(indices)].reset_index()
        self.tiles = dataset.tiles.query(f"slide_id in {tuple(self.slides['id'])}")


class TileEmbeddingsSubset(Subset[TileEmbeddingsSample]):
    def __init__(
        self,
        dataset: TileEmbeddings,
        indices: Sequence[int],
    ) -> None:
        super().__init__(dataset, indices)
        self.slides = dataset.slides.iloc[list(indices)].reset_index()
        self.tiles = dataset.tiles.query(f"slide_id in {tuple(self.slides['id'])}")


class SlideEmbeddingsSubset(Subset[SlideEmbeddingsSample]):
    def __init__(
        self,
        dataset: SlideEmbeddings,
        indices: Sequence[int],
    ) -> None:
        super().__init__(dataset, indices)
        self.slides = dataset.slides.iloc[list(indices)].reset_index()


def create_subset(
    dataset: Tiles | TileEmbeddings | SlideEmbeddings, indices: Sequence[int]
) -> TilesSubset | TileEmbeddingsSubset | SlideEmbeddingsSubset:
    if isinstance(dataset, Tiles):
        return TilesSubset(dataset, indices)
    if isinstance(dataset, TileEmbeddings):
        return TileEmbeddingsSubset(dataset, indices)
    if isinstance(dataset, SlideEmbeddings):
        return SlideEmbeddingsSubset(dataset, indices)

    raise TypeError(f"Unsupported dataset type: {type(dataset)}")
