from collections.abc import Iterable

import pandas as pd
import torch
import torch.nn.functional as F
from albumentations import TransformType
from albumentations.pytorch import ToTensorV2
from rationai.mlkit.data.datasets import MetaTiledSlides, OpenSlideTilesDataset
from torch.utils.data import Dataset

from ulcerative_colitis.typing import Metadata, Sample


class UlcerativeColitis(MetaTiledSlides[Sample]):
    def __init__(
        self,
        uris: Iterable[str],
        transforms: TransformType | None = None,
    ) -> None:
        self.transforms = transforms
        super().__init__(uris=uris)

    def generate_datasets(self) -> Iterable[Dataset[Sample]]:
        return (
            _UlcerativeColitisSlideTiles(
                slide,
                tiles=self.filter_tiles_by_slide(slide["id"]),
                transforms=self.transforms,
            )
            for _, slide in self.slides.iterrows()
        )


class _UlcerativeColitisSlideTiles(Dataset[Sample]):
    def __init__(
        self,
        slide_metadata: pd.Series,
        tiles: pd.DataFrame,
        transforms: TransformType | None = None,
    ) -> None:
        super().__init__()
        self.slide_tiles = OpenSlideTilesDataset(
            slide_path=slide_metadata.path,
            level=slide_metadata.level,
            tile_extent_x=slide_metadata.tile_extent_x,
            tile_extent_y=slide_metadata.tile_extent_y,
            tiles=tiles,
        )
        self.transforms = transforms
        self.to_tensor = ToTensorV2()

    def __len__(self) -> int:
        return len(self.slide_tiles)

    def __getitem__(self, idx: int) -> Sample:
        image = self.slide_tiles[idx]
        metadata = Metadata(
            slide=self.slide_tiles.slide_path.stem,
            x=self.slide_tiles.tiles.iloc[idx]["x"],
            y=self.slide_tiles.tiles.iloc[idx]["y"],
        )

        if self.transforms is not None:
            image = self.transforms(image=image)["image"]

        image = self.to_tensor(image=image)["image"]

        label_index = torch.tensor([self.slide_tiles.tiles.iloc[idx]["nancy_index"]])
        label = F.one_hot(label_index, num_classes=5).float()
        return image, label, metadata
