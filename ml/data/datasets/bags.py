from collections.abc import Iterable
from pathlib import Path
from typing import Any, Generic, TypeVar

import torch.nn.functional as F
from datasets import Dataset as HFDataset
from rationai.mlkit.data.datasets import SlidesTilesLoader
from torch import Tensor
from torch.utils.data import Dataset

from ml.data.datasets.labels import LabelMode, get_label, process_slides
from ml.data.datasets.utils import (
    column_tensor,
    download_artifacts_cached,
    embeddings_tensor,
    filter_tiles,
)
from ml.typing import BagsPredictSample, BagsSample, MetadataBag


T = TypeVar("T", BagsSample, BagsPredictSample)


class _Bags(Dataset[T], Generic[T]):
    def __init__(
        self,
        uris: Iterable[str] | str,
        mode: LabelMode | None,
        padding: bool = True,
        thresholds: dict[str, float] | None = None,
        cache_dir: Path | str | None = None,
    ) -> None:
        self.mode = mode
        self.thresholds = thresholds or {}

        uris = [uris] if isinstance(uris, str) else list(uris)
        if cache_dir is None:
            self._meta = SlidesTilesLoader(uris=uris)
        else:
            cache_dir = Path(cache_dir)
            self._meta = SlidesTilesLoader(
                paths=[
                    download_artifacts_cached(uri, cache_dir / "mlflow") for uri in uris
                ],
                hf_kwargs={
                    "path": "parquet",
                    "split": "train",
                    "cache_dir": str(cache_dir / "huggingface"),
                },
            )
        self.tiles = self._meta.tiles
        if self.thresholds:
            self.tiles = filter_tiles(self.tiles, self.thresholds)
            self._meta.tiles = self.tiles
            self._meta._slide_id_to_indices = self._meta._build_tile_index(self.tiles)

        # slides without any tile would produce an empty bag
        slide_ids = set(self._meta._slide_id_to_indices)
        self.slides = process_slides(
            self._meta.slides.filter(lambda slide: slide["id"] in slide_ids),
            self.mode,
        )

        self.padding = padding
        self.max_embeddings = max(
            len(indices) for indices in self._meta._slide_id_to_indices.values()
        )

    def __len__(self) -> int:
        return len(self.slides)

    def _bag(self, slide_metadata: dict[str, Any]) -> tuple[Tensor, MetadataBag]:
        tiles: HFDataset = self._meta.filter_tiles_by_slide(slide_metadata["id"])
        embeddings = embeddings_tensor(tiles)

        pad_amount = self.max_embeddings - embeddings.shape[0]
        if self.padding:
            embeddings = F.pad(embeddings, (0, 0, 0, pad_amount), value=0.0)

        metadata = MetadataBag(
            slide_name=str(slide_metadata["name"]),
            slide_path=Path(slide_metadata["path"]),
            level=slide_metadata["level"],
            tile_extent_x=slide_metadata["tile_extent_x"],
            tile_extent_y=slide_metadata["tile_extent_y"],
            tiles=tiles,
            x=column_tensor(tiles, "x"),
            y=column_tensor(tiles, "y"),
        )
        return embeddings, metadata


class Bags(_Bags[BagsSample]):
    mode: LabelMode

    def __init__(
        self,
        uris: Iterable[str] | str,
        mode: LabelMode | str,
        padding: bool = True,
        thresholds: dict[str, float] | None = None,
        cache_dir: Path | str | None = None,
    ) -> None:
        super().__init__(uris, LabelMode(mode), padding, thresholds, cache_dir)

    @property
    def labels(self) -> list[int]:
        return [int(get_label(dict(slide), self.mode).item()) for slide in self.slides]

    def __getitem__(self, idx: int) -> BagsSample:
        slide_metadata = self.slides[idx]
        embeddings, metadata = self._bag(slide_metadata)
        return embeddings, get_label(slide_metadata, self.mode), metadata


class BagsPredict(_Bags[BagsPredictSample]):
    def __init__(
        self,
        uris: Iterable[str] | str,
        padding: bool = True,
        thresholds: dict[str, float] | None = None,
        cache_dir: Path | str | None = None,
    ) -> None:
        super().__init__(uris, None, padding, thresholds, cache_dir)

    def __getitem__(self, idx: int) -> BagsPredictSample:
        return self._bag(self.slides[idx])
