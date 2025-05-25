from collections.abc import Iterable

import pandas as pd
import torch
from albumentations.core.composition import TransformType
from torch.utils.data import Dataset

from ulcerative_colitis.data.datasets.ulcerative_colitis import (
    UlcerativeColitisPredict,
    UlcerativeColitisSlidesTilesTrain,
    UlcerativeColitisSlideTiles,
    UlcerativeColitisTest,
    UlcerativeColitisTrain,
)
from ulcerative_colitis.typing import PredictSample, TestSample, TrainSample


# LOCATIONS = ("cekoascendens", "descendens", "rektosigma", "transverzum")


class NeutrophilsTrain(UlcerativeColitisTrain):
    def generate_datasets(self) -> Iterable[Dataset[TrainSample]]:
        self.slides = process_slides(self.slides)
        return [
            _NeutrophilsSlidesTilesTrain(
                self.slides,
                self.tiles,
                self.inner_batch_size,
                self.transforms,
            )
        ]


class NeutrophilsTest(UlcerativeColitisTest):
    def generate_datasets(self) -> Iterable[Dataset[TestSample]]:
        self.slides = process_slides(self.slides)
        return [
            _NeutrophilsSlideTilesTest(
                slide,
                tiles=self.filter_tiles_by_slide(slide["id"]),
                transforms=self.transforms,
            )
            for _, slide in self.slides.iterrows()
        ]


class NeutrophilsPredict(UlcerativeColitisPredict):
    def generate_datasets(self) -> Iterable[Dataset[PredictSample]]:
        self.slides = process_slides(self.slides)
        return [
            _NetrophilsSlideTilesPredict(
                slide,
                tiles=self.filter_tiles_by_slide(slide["id"]),
                transforms=self.transforms,
            )
            for _, slide in self.slides.iterrows()
        ]


class _NeutrophilsSlidesTilesTrain(UlcerativeColitisSlidesTilesTrain):
    def get_label(self, slide_metadata: pd.Series) -> torch.Tensor:
        return torch.tensor([slide_metadata["neutrophils"]]).float()


class _NeutrophilsSlideTilesTest(UlcerativeColitisSlideTiles):
    def __init__(
        self,
        slide_metadata: pd.Series,
        tiles: pd.DataFrame,
        transforms: TransformType | None = None,
    ) -> None:
        super().__init__(slide_metadata, tiles, True, transforms)

    def get_label(self, slide_metadata: pd.Series) -> torch.Tensor:
        return torch.tensor([slide_metadata["neutrophils"]]).float()


class _NetrophilsSlideTilesPredict(UlcerativeColitisSlideTiles):
    def __init__(
        self,
        slide_metadata: pd.Series,
        tiles: pd.DataFrame,
        transforms: TransformType | None = None,
    ) -> None:
        super().__init__(slide_metadata, tiles, False, transforms)


def process_slides(slides: pd.DataFrame) -> pd.DataFrame:
    # slides = slides[slides["location"].isin(LOCATIONS)].copy()
    slides["neutrophils"] = slides["nancy_index"] >= 2
    return slides
