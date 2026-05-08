from collections.abc import Iterable
from typing import Generic, TypeVar

import torch
from datasets import Dataset as HFDataset
from rationai.mlkit.data.datasets import MetaTiledSlides
from torch.utils.data import Dataset

from ml.data.datasets.labels import LabelMode, get_label, process_slides
from ml.data.datasets.utils import filter_tiles
from ml.typing import EmbeddingsPredictSample, EmbeddingsSample, MetadataEmbeddings


T = TypeVar("T", bound=EmbeddingsSample | EmbeddingsPredictSample)


class _Embeddings(Generic[T], Dataset[T]):
    def __init__(
        self,
        slide_metadata: dict,
        tiles: HFDataset,
        mode: LabelMode | str | None,
        include_labels: bool = True,
    ) -> None:
        self.slide_metadata = slide_metadata
        self.tiles = tiles
        self.mode = LabelMode(mode) if mode is not None else None
        self.include_labels = include_labels

        if self.include_labels and self.mode is None:
            raise ValueError("Mode must be specified if labels are included.")

    def __len__(self) -> int:
        return len(self.tiles)

    def __getitem__(self, idx: int) -> EmbeddingsSample | EmbeddingsPredictSample:
        tile = self.tiles[idx]
        metadata = MetadataEmbeddings(
            slide_name=self.slide_metadata["name"],
            x=tile["x"],
            y=tile["y"],
        )

        embedding = torch.tensor(tile["embedding"])
        if not self.include_labels:
            return embedding, metadata

        assert self.mode is not None, "Mode must be specified for labels."
        label = get_label(tile, self.mode)
        return embedding, label, metadata


class Embeddings(MetaTiledSlides[EmbeddingsSample]):
    def __init__(
        self,
        uris: Iterable[str] | str,
        mode: LabelMode | str,
        thresholds: dict[str, float] | None = None,
        val_fold: int | None = None,
        is_val: bool = False,
    ) -> None:
        self.mode = LabelMode(mode)
        self.thresholds = thresholds or {}
        self.val_fold = val_fold
        self.is_val = is_val
        super().__init__(uris=(uris,) if isinstance(uris, str) else uris)

    def generate_datasets(self) -> Iterable[_Embeddings[EmbeddingsSample]]:
        self.slides = process_slides(self.slides, self.mode, val_fold=self.val_fold, is_val=self.is_val)
        return (
            _Embeddings(
                slide_metadata=dict(slide),
                tiles=filter_tiles(
                    self.filter_tiles_by_slide(dict(slide)["id"]), self.thresholds
                ),
                mode=self.mode,
                include_labels=True,
            )
            for slide in self.slides
        )


class EmbeddingsPredict(MetaTiledSlides[EmbeddingsPredictSample]):
    def __init__(
        self,
        uris: Iterable[str] | str,
        mode: LabelMode | str,
        thresholds: dict[str, float] | None = None,
    ) -> None:
        self.mode = LabelMode(mode)
        self.thresholds = thresholds or {}
        super().__init__(uris=(uris,) if isinstance(uris, str) else uris)

    def generate_datasets(self) -> Iterable[_Embeddings[EmbeddingsPredictSample]]:
        self.slides = process_slides(self.slides, self.mode)
        return (
            _Embeddings(
                slide_metadata=dict(slide),
                tiles=filter_tiles(
                    self.filter_tiles_by_slide(dict(slide)["id"]), self.thresholds
                ),
                mode=self.mode,
                include_labels=False,
            )
            for slide in self.slides
        )
