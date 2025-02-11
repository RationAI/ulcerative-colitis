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


LOCATIONS = ("cekoascendens", "descendens", "rektosigma", "transverzum")


class NancyHighTrain(UlcerativeColitisTrain):
    def generate_datasets(self) -> Iterable[Dataset[TrainSample]]:
        self.slides = process_slides(self.slides)
        return [
            _NancyHighSlidesTilesTrain(
                self.slides,
                self.tiles,
                self.inner_batch_size,
                self.transforms,
            )
        ]


class NancyHighTest(UlcerativeColitisTest):
    def generate_datasets(self) -> Iterable[Dataset[TestSample]]:
        self.slides = process_slides(self.slides)
        return [
            _NancyHighSlideTilesTest(
                slide,
                tiles=self.filter_tiles_by_slide(slide["id"]),
                transforms=self.transforms,
            )
            for _, slide in self.slides.iterrows()
        ]


class NancyHighPredict(UlcerativeColitisPredict):
    def generate_datasets(self) -> Iterable[Dataset[PredictSample]]:
        self.slides = process_slides(self.slides)
        return [
            _NancyHighSlideTilesPredict(
                slide,
                tiles=self.filter_tiles_by_slide(slide["id"]),
                transforms=self.transforms,
            )
            for _, slide in self.slides.iterrows()
        ]


class _NancyHighSlidesTilesTrain(UlcerativeColitisSlidesTilesTrain):
    def get_label(self, slide_metadata: pd.Series) -> torch.Tensor:
        return torch.tensor(slide_metadata["nancy_index"]).float()


class _NancyHighSlideTilesTest(UlcerativeColitisSlideTiles):
    def __init__(
        self,
        slide_metadata: pd.Series,
        tiles: pd.DataFrame,
        transforms: TransformType | None = None,
    ) -> None:
        super().__init__(slide_metadata, tiles, True, transforms)

    def get_label(self, slide_metadata: pd.Series) -> torch.Tensor:
        return torch.tensor(slide_metadata["nancy_index"]).float()


class _NancyHighSlideTilesPredict(UlcerativeColitisSlideTiles):
    def __init__(
        self,
        slide_metadata: pd.Series,
        tiles: pd.DataFrame,
        transforms: TransformType | None = None,
    ) -> None:
        super().__init__(slide_metadata, tiles, False, transforms)


def process_slides(slides: pd.DataFrame) -> pd.DataFrame:
    slides = slides[slides["location"].isin(LOCATIONS)]
    return slides[slides["nancy_index"] >= 2]
