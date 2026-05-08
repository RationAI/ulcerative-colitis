from collections.abc import Iterable
from typing import Generic, TypeVar

from albumentations.core.composition import TransformType
from albumentations.pytorch import ToTensorV2
from datasets import Dataset as HFDataset
from rationai.mlkit.data.datasets import MetaTiledSlides, OpenSlideTilesDataset
from torch.utils.data import Dataset

from ml.data.datasets.labels import LabelMode, get_label, process_slides
from ml.data.datasets.utils import filter_tiles
from ml.typing import MetadataTiles, TilesPredictSample, TilesSample


T = TypeVar("T", bound=TilesSample | TilesPredictSample)


class _Tiles(Dataset[T], Generic[T]):
    def __init__(
        self,
        slide_metadata: dict,
        tiles: HFDataset,
        mode: LabelMode | str | None,
        include_labels: bool = True,
        transforms: TransformType | None = None,
        to_tensor: bool = True,
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
        self.to_tensor = ToTensorV2() if to_tensor else None

        if self.include_labels and self.mode is None:
            raise ValueError("Mode must be specified if labels are included.")

    def __len__(self) -> int:
        return len(self.slide_tiles)

    def __getitem__(self, idx: int) -> TilesSample | TilesPredictSample:
        image = self.slide_tiles[idx]
        metadata = MetadataTiles(
            slide_name=self.slide_tiles.slide_path.stem,
            x=self.slide_tiles.tiles[idx]["x"],
            y=self.slide_tiles.tiles[idx]["y"],
        )

        if self.transforms is not None:
            image = self.transforms(image=image)["image"]

        if self.to_tensor is not None:
            image = self.to_tensor(image=image)["image"]

        if not self.include_labels:
            return image, metadata

        assert self.mode is not None, "Mode must be specified for labels."
        label = get_label(self.slide_metadata, self.mode)
        return image, label, metadata


class Tiles(MetaTiledSlides[TilesSample]):
    @property
    def labels(self) -> list[int]:
        return [
            int(get_label(ds.slide_metadata, self.mode).item())
            for ds in self.datasets
            for _ in range(len(ds))
        ]

    def __init__(
        self,
        uris: Iterable[str] | str,
        mode: LabelMode | str,
        transforms: TransformType | None = None,
        to_tensor: bool = True,
        thresholds: dict[str, float] | None = None,
        val_fold: int | None = None,
        is_val: bool = False,
    ) -> None:
        self.transforms = transforms
        self.mode = LabelMode(mode)
        self.to_tensor = to_tensor
        self.thresholds = thresholds or {}
        self.val_fold = val_fold
        self.is_val = is_val
        super().__init__(uris=(uris,) if isinstance(uris, str) else uris)

    def generate_datasets(self) -> Iterable[_Tiles[TilesSample]]:
        self.slides = process_slides(self.slides, self.mode, val_fold=self.val_fold, is_val=self.is_val)
        return (
            _Tiles(
                slide_metadata=dict(slide),
                tiles=filter_tiles(
                    self.filter_tiles_by_slide(dict(slide)["id"]), self.thresholds
                ),
                mode=self.mode,
                include_labels=True,
                transforms=self.transforms,
                to_tensor=self.to_tensor,
            )
            for slide in self.slides
        )


class TilesPredict(MetaTiledSlides[TilesPredictSample]):
    def __init__(
        self,
        uris: Iterable[str] | str,
        transforms: TransformType | None = None,
        to_tensor: bool = True,
        thresholds: dict[str, float] | None = None,
    ) -> None:
        self.transforms = transforms
        self.to_tensor = to_tensor
        self.thresholds = thresholds or {}
        super().__init__(uris=(uris,) if isinstance(uris, str) else uris)

    def generate_datasets(self) -> Iterable[_Tiles[TilesPredictSample]]:
        self.slides = process_slides(self.slides)
        return (
            _Tiles(
                slide_metadata=dict(slide),
                tiles=filter_tiles(
                    self.filter_tiles_by_slide(dict(slide)["id"]), self.thresholds
                ),
                mode=None,
                include_labels=False,
                transforms=self.transforms,
                to_tensor=self.to_tensor,
            )
            for slide in self.slides
        )
