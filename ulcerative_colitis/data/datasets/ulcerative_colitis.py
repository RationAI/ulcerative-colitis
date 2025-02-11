import random
from collections.abc import Iterable
from typing import Generic, TypeVar

import pandas as pd
import torch
from albumentations.core.composition import TransformType
from albumentations.pytorch import ToTensorV2
from rationai.mlkit.data.datasets import MetaTiledSlides, OpenSlideTilesDataset
from torch.utils.data import Dataset

from ulcerative_colitis.typing import (
    Metadata,
    PredictSample,
    TestSample,
    TrainMetadata,
    TrainSample,
)


T = TypeVar("T", bound=TestSample | PredictSample)


class UlcerativeColitisTrain(MetaTiledSlides[TrainSample]):
    def __init__(
        self,
        uris: Iterable[str],
        inner_batch_size: int,
        transforms: TransformType | None = None,
    ) -> None:
        self.inner_batch_size = inner_batch_size
        self.transforms = transforms
        super().__init__(uris=uris)


class UlcerativeColitisTest(MetaTiledSlides[TestSample]):
    def __init__(
        self,
        uris: Iterable[str],
        transforms: TransformType | None = None,
    ) -> None:
        self.transforms = transforms
        super().__init__(uris=uris)


class UlcerativeColitisPredict(MetaTiledSlides[PredictSample]):
    def __init__(
        self,
        uris: Iterable[str],
        transforms: TransformType | None = None,
    ) -> None:
        self.transforms = transforms
        super().__init__(uris=uris)


class UlcerativeColitisSlideTiles(Dataset[T], Generic[T]):
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

    def __getitem__(self, idx: int) -> TestSample | PredictSample:
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

        label = self.get_label(self.slide_metadata)
        return image, label, metadata

    def get_label(self, slide_metadata: pd.Series) -> torch.Tensor:
        raise NotImplementedError


class UlcerativeColitisSlidesTilesTrain(Dataset[TrainSample]):
    def __init__(
        self,
        slides: pd.DataFrame,
        tiles: pd.DataFrame,
        inner_batch_size: int,
        transforms: TransformType | None = None,
    ) -> None:
        super().__init__()
        self.slides_tiles = [
            OpenSlideTilesDataset(
                slide_path=slide["path"],
                level=slide["level"],
                tile_extent_x=slide["tile_extent_x"],
                tile_extent_y=slide["tile_extent_y"],
                tiles=tiles[tiles["slide_id"] == slide["id"]],
            )
            for _, slide in slides.iterrows()
        ]

        self.slides = slides
        self.tiles = tiles
        self.inner_batch_size = inner_batch_size
        self.transforms = transforms
        self.to_tensor = ToTensorV2()

    def __len__(self) -> int:
        return len(self.slides_tiles)

    def __getitem__(self, idx: int) -> TrainSample:
        slide_tiles = self.slides_tiles[idx]
        slide_metadata = self.slides.iloc[idx]

        images = []
        xs = []
        ys = []
        for _ in range(self.inner_batch_size):
            i = random.randint(0, len(slide_tiles) - 1)
            image = slide_tiles[i]
            if self.transforms is not None:
                image = self.transforms(image=image)["image"]

            image = self.to_tensor(image=image)["image"]
            images.append(image)

            xs.append(slide_tiles.tiles.iloc[i]["x"])
            ys.append(slide_tiles.tiles.iloc[i]["y"])

        label = self.get_label(slide_metadata)
        metadata = TrainMetadata(
            slide=[slide_tiles.slide_path.stem] * self.inner_batch_size,
            x=torch.tensor(xs),
            y=torch.tensor(ys),
        )

        return torch.stack(images), label, metadata

    def get_label(self, slide_metadata: pd.Series) -> torch.Tensor:
        raise NotImplementedError
