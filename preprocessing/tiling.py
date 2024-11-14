from dataclasses import asdict, dataclass
from pathlib import Path

import mlflow
import pandas as pd
import ray
from rationai.tiling import tiling
from rationai.tiling.modules.masks import PyvipsMask
from rationai.tiling.modules.tile_sources import OpenSlideTileSource
from rationai.tiling.typing import TiledSlideMetadata, TileMetadata
from rationai.tiling.writers import save_mlflow_dataset
from sklearn.model_selection import train_test_split


SLIDES_PATH = Path(
    "/mnt/data/Projects/inflammatory_bowel_dissease/ulcerative_colitis/tiff/"
)
DATAFRAME_PATH = SLIDES_PATH.parent / "test_cohort" / "IBD_AI_test_Fabian.csv"
TISSUE_MASKS_PATH = Path("data/tissue_masks")


@dataclass
class NancyIndexTileMetadata(TileMetadata):
    nancy_index: int


class TissueMask(PyvipsMask[TileMetadata]):
    def forward_tile(
        self, tile_labels: TileMetadata, class_overlaps: dict[int, float]
    ) -> TileMetadata | None:
        if class_overlaps.get(0, 0) > 0.5:
            return None
        return tile_labels


source = OpenSlideTileSource(mpp=0.48, tile_extent=512, stride=256)
tissue_mask = TissueMask(
    tile_extent=source.tile_extent, absolute_roi_extent=256, relative_roi_offset=0
)
df = pd.read_csv(DATAFRAME_PATH, index_col=0)


def train_test_split_cases(
    slides: list[Path], test_size: float, random_state: int = 42
) -> tuple[list[Path], list[Path]]:
    cases = set()
    for slide in slides:
        cases.add(slide.stem[:7])

    train_cases, test_cases = train_test_split(
        list(cases), test_size=test_size, random_state=random_state
    )

    return (
        [slide for slide in slides if slide.stem[:7] in train_cases],
        [slide for slide in slides if slide.stem[:7] in test_cases],
    )


def get_nancy_index(slide_path: Path, df_nancy_index: pd.DataFrame) -> int:
    stem = slide_path.stem
    index = f"{stem[:4]}/{stem[5:7]}"
    return int(pd.to_numeric(df_nancy_index.loc[index, "Nancy"]))


@ray.remote
def handler(slide_path: Path) -> TiledSlideMetadata:
    slide, tiles = source(slide_path)

    tissue_mask_path = TISSUE_MASKS_PATH / slide_path.name

    tiles = tissue_mask(tissue_mask_path, slide.extent, tiles)
    tiles = [
        NancyIndexTileMetadata(
            **asdict(tile), nancy_index=get_nancy_index(slide_path, df)
        )
        for tile in tiles
    ]

    return slide, tiles


def main() -> None:
    slides, test_slides = train_test_split(
        list(SLIDES_PATH.rglob("*.tiff")), test_size=0.2
    )
    train_slides, val_slides = train_test_split(slides, test_size=0.1)

    train_slides_df, train_tiles_df = tiling(slides=train_slides, handler=handler)
    val_slides_df, val_tiles_df = tiling(slides=val_slides, handler=handler)
    test_slides_df, test_tiles_df = tiling(slides=test_slides, handler=handler)

    mlflow.set_experiment(experiment_name="IKEM")
    with mlflow.start_run(run_name="📂 Dataset: Ulcerative Colitis"):
        save_mlflow_dataset(
            slides=train_slides_df,
            tiles=train_tiles_df,
            dataset_name="Ulcerative Colitis - train",
        )
        save_mlflow_dataset(
            slides=val_slides_df,
            tiles=val_tiles_df,
            dataset_name="Ulcerative Colitis - val",
        )
        save_mlflow_dataset(
            slides=test_slides_df,
            tiles=test_tiles_df,
            dataset_name="Ulcerative Colitis - test",
        )


if __name__ == "__main__":
    main()
