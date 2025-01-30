from collections.abc import Iterable
from typing import Generic, TypeVar

import pandas as pd
import torch
from albumentations.core.composition import TransformType
from albumentations.pytorch import ToTensorV2
from rationai.mlkit.data.datasets import MetaTiledSlides, OpenSlideTilesDataset
from torch.utils.data import Dataset

from ulcerative_colitis.typing import Metadata, PredictSample, Sample


T = TypeVar("T", bound=Sample | PredictSample)
LOCATIONS = ("cekoascendens", "descendens", "rektosigma", "transverzum")


class UlcerativeColitis(MetaTiledSlides[Sample]):
    def __init__(
        self,
        uris: Iterable[str],
        transforms: TransformType | None = None,
    ) -> None:
        self.transforms = transforms
        super().__init__(uris=uris)

    def generate_datasets(self) -> Iterable[Dataset[Sample]]:
        self.slides = self.slides[self.slides["location"].isin(LOCATIONS)]
        return (
            _UlcerativeColitisSlideTiles[Sample](
                slide,
                tiles=self.filter_tiles_by_slide(slide["id"]),
                transforms=self.transforms,
            )
            for _, slide in self.slides.iterrows()
        )


class UlcerativeColitisPredict(MetaTiledSlides[PredictSample]):
    def __init__(
        self,
        uris: Iterable[str],
        transforms: TransformType | None = None,
    ) -> None:
        self.transforms = transforms
        super().__init__(uris=uris)

    def generate_datasets(self) -> Iterable[Dataset[PredictSample]]:
        self.slides = self.slides[self.slides["location"].isin(LOCATIONS)]
        return (
            _UlcerativeColitisSlideTiles[PredictSample](
                slide,
                tiles=self.filter_tiles_by_slide(slide["id"]),
                include_labels=False,
                transforms=self.transforms,
            )
            for _, slide in self.slides.iterrows()
        )


class _UlcerativeColitisSlideTiles(Dataset[T], Generic[T]):
    def __init__(
        self,
        slide_metadata: pd.Series,
        tiles: pd.DataFrame,
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
        self.include_labels = include_labels
        self.transforms = transforms
        self.to_tensor = ToTensorV2()

    def __len__(self) -> int:
        return len(self.slide_tiles)

    def __getitem__(self, idx: int) -> Sample | PredictSample:
        image = self.slide_tiles[idx]
        metadata = Metadata(
            slide=self.slide_tiles.slide_path.stem,
            x=self.slide_tiles.tiles.iloc[idx]["x"],
            y=self.slide_tiles.tiles.iloc[idx]["y"],
        )

        if self.transforms is not None:
            image = self.transforms(image=image)["image"]

        image = self.to_tensor(image=image)["image"]
        if not self.include_labels:
            return image, metadata

        label = torch.tensor([self.slide_metadata["nancy_index"]])
        return image, label, metadata
