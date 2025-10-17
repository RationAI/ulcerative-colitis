from collections.abc import Iterable
from typing import Generic, TypeVar

import pandas as pd
from albumentations.core.composition import TransformType
from albumentations.pytorch import ToTensorV2
from rationai.mlkit.data.datasets import MetaTiledSlides, OpenSlideTilesDataset
from torch.utils.data import Dataset

from ulcerative_colitis.data.datasets.labels import LabelMode, get_label, process_slides
from ulcerative_colitis.typing import MetadataTiles, TilesPredictSample, TilesSample


T = TypeVar("T", bound=TilesSample | TilesPredictSample)


class _Tiles(Dataset[T], Generic[T]):
    def __init__(
        self,
        slide_metadata: pd.Series,
        tiles: pd.DataFrame,
        mode: LabelMode | str | None,
        include_labels: bool = True,
        transforms: TransformType | None = None,
    ) -> None:
        super().__init__()
        self.slide_tiles = OpenSlideTilesDataset(
            slide_path=slide_metadata["path"],
            level=slide_metadata["level"],
            tile_extent_x=slide_metadata["tile_extent_x"],
            tile_extent_y=slide_metadata["tile_extent_y"],
            tiles=tiles,
        )
        self.slide_metadata = slide_metadata
        self.mode = LabelMode(mode) if mode is not None else None
        self.include_labels = include_labels
        self.transforms = transforms
        self.to_tensor = ToTensorV2()

        if self.include_labels and self.mode is None:
            raise ValueError("Mode must be specified if labels are included.")

    def __len__(self) -> int:
        return len(self.slide_tiles)

    def __getitem__(self, idx: int) -> TilesSample | TilesPredictSample:
        image = self.slide_tiles[idx]
        metadata = MetadataTiles(
            slide_id=self.slide_tiles.slide_path.stem,
            x=self.slide_tiles.tiles.iloc[idx]["x"],
            y=self.slide_tiles.tiles.iloc[idx]["y"],
        )

        if self.transforms is not None:
            image = self.transforms(image=image)["image"]

        image = self.to_tensor(image=image)["image"]
        if not self.include_labels:
            return image, metadata

        assert self.mode is not None, "Mode must be specified for labels."
        label = get_label(self.slide_metadata, self.mode)
        return image, label, metadata


class Tiles(MetaTiledSlides[TilesSample]):
    def __init__(
        self,
        uris: Iterable[str],
        mode: LabelMode | str,
        transforms: TransformType | None = None,
    ) -> None:
        self.transforms = transforms
        self.mode = LabelMode(mode)
        super().__init__(uris=uris)

    def generate_datasets(self) -> Iterable[_Tiles[TilesSample]]:
        self.slides = process_slides(self.slides, self.mode)
        return (
            _Tiles(
                slide_metadata=slide,
                tiles=self.filter_tiles_by_slide(slide["id"]),
                mode=self.mode,
                include_labels=True,
                transforms=self.transforms,
            )
            for _, slide in self.slides.iterrows()
        )


class TilesPredict(MetaTiledSlides[TilesPredictSample]):
    def __init__(
        self,
        uris: Iterable[str],
        transforms: TransformType | None = None,
    ) -> None:
        self.transforms = transforms
        super().__init__(uris=uris)

    def generate_datasets(self) -> Iterable[_Tiles[TilesPredictSample]]:
        self.slides = process_slides(self.slides)
        return (
            _Tiles(
                slide_metadata=slide,
                tiles=self.filter_tiles_by_slide(slide["id"]),
                mode=None,
                include_labels=False,
                transforms=self.transforms,
            )
            for _, slide in self.slides.iterrows()
        )
