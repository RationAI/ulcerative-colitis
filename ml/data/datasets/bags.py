from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import Generic, TypeVar

import torch
import torch.nn.functional as F
from rationai.mlkit.data.datasets import SlidesTilesLoader
from torch.utils.data import Dataset

from ml.data.datasets.labels import LabelMode, get_label, process_slides
from ml.data.datasets.utils import filter_tiles
from ml.typing import BagsPredictSample, BagsSample, MetadataBags


T = TypeVar("T", bound=BagsSample | BagsPredictSample)


class _Bags(Dataset[T], Generic[T]):
    def __init__(
        self,
        uris: Iterable[str] | str,
        mode: LabelMode | str | None = None,
        padding: bool = True,
        include_labels: bool = True,
        thresholds: dict[str, float] | None = None,
        val_fold: int | None = None,
        is_val: bool = False,
    ) -> None:
        self.mode = LabelMode(mode) if mode is not None else None
        self.include_labels = include_labels
        self.thresholds = thresholds or {}

        if self.include_labels and self.mode is None:
            raise ValueError("Mode must be specified when including labels.")

        self._meta = SlidesTilesLoader(uris=uris)
        self.tiles = filter_tiles(self._meta.tiles, self.thresholds)
        self._meta.tiles = self.tiles
        self._meta._slide_id_to_indices = self._meta._build_tile_index(self.tiles)
        self.slides = process_slides(
            self._meta.slides, self.mode, val_fold=val_fold, is_val=is_val
        )

        self.padding = padding
        self.max_embeddings = max(Counter(self.tiles["slide_id"]).values())

    def __len__(self) -> int:
        return len(self.slides)

    def __getitem__(self, idx: int) -> T:
        slide_metadata = self.slides[idx]
        tiles = self._meta.filter_tiles_by_slide(slide_metadata["id"])
        slide_name = str(slide_metadata["name"])
        embeddings = torch.tensor(tiles["embedding"])

        pad_amount = self.max_embeddings - embeddings.shape[0]
        if self.padding:
            embeddings = F.pad(embeddings, (0, 0, 0, pad_amount), value=0.0)

        metadata = MetadataBags(
            slide_name=slide_name,
            slide_path=Path(slide_metadata["path"]),
            level=slide_metadata["level"],
            tile_extent_x=slide_metadata["tile_extent_x"],
            tile_extent_y=slide_metadata["tile_extent_y"],
            tiles=tiles,
            x=torch.tensor(tiles["x"]),
            y=torch.tensor(tiles["y"]),
        )

        if not self.include_labels:
            return embeddings, metadata

        assert self.mode is not None
        label = get_label(slide_metadata, self.mode)

        return embeddings, label, metadata


class Bags(_Bags[BagsSample]):
    @property
    def labels(self) -> list[int]:
        assert self.mode is not None
        return [int(get_label(dict(slide), self.mode).item()) for slide in self.slides]

    def __init__(
        self,
        uris: Iterable[str] | str,
        mode: LabelMode | str,
        padding: bool = True,
        thresholds: dict[str, float] | None = None,
        val_fold: int | None = None,
        is_val: bool = False,
    ) -> None:
        super().__init__(
            uris=uris,
            mode=mode,
            padding=padding,
            include_labels=True,
            thresholds=thresholds,
            val_fold=val_fold,
            is_val=is_val,
        )


class BagsPredict(_Bags[BagsPredictSample]):
    def __init__(
        self,
        uris: Iterable[str] | str,
        mode: LabelMode | str | None = None,
        padding: bool = True,
        thresholds: dict[str, float] | None = None,
    ) -> None:
        super().__init__(
            uris=uris,
            mode=mode,
            padding=padding,
            include_labels=False,
            thresholds=thresholds,
        )
