from collections import Counter
from collections.abc import Iterable
from pathlib import Path

import torch
import torch.nn.functional as F
from rationai.mlkit.data.datasets import SlidesTilesLoader
from torch.utils.data import Dataset

from ml.data.datasets.labels import LabelMode, get_label, process_slides
from ml.data.datasets.utils import filter_tiles
from ml.typing import BagsSample, MetadataBags


class Bags(Dataset[BagsSample]):
    def __init__(
        self,
        uris: Iterable[str] | str,
        mode: LabelMode | str,
        padding: bool = True,
        thresholds: dict[str, float] | None = None,
    ) -> None:
        self.mode = LabelMode(mode)
        self.thresholds = thresholds or {}

        self._meta = SlidesTilesLoader(uris=[uris] if isinstance(uris, str) else uris)
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
        self.max_embeddings = max(Counter(self.tiles["slide_id"]).values())

    @property
    def labels(self) -> list[int]:
        return [int(get_label(dict(slide), self.mode).item()) for slide in self.slides]

    def __len__(self) -> int:
        return len(self.slides)

    def __getitem__(self, idx: int) -> BagsSample:
        slide_metadata = self.slides[idx]
        tiles = self._meta.filter_tiles_by_slide(slide_metadata["id"])
        embeddings = torch.tensor(tiles["embedding"])

        pad_amount = self.max_embeddings - embeddings.shape[0]
        if self.padding:
            embeddings = F.pad(embeddings, (0, 0, 0, pad_amount), value=0.0)

        metadata = MetadataBags(
            slide_name=str(slide_metadata["name"]),
            slide_path=Path(slide_metadata["path"]),
            level=slide_metadata["level"],
            tile_extent_x=slide_metadata["tile_extent_x"],
            tile_extent_y=slide_metadata["tile_extent_y"],
            tiles=tiles,
            x=torch.tensor(tiles["x"]),
            y=torch.tensor(tiles["y"]),
        )

        return embeddings, get_label(slide_metadata, self.mode), metadata
