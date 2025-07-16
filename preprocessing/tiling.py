from collections.abc import Iterable
from dataclasses import asdict
from pathlib import Path
from typing import Any

import mlflow
import pandas as pd
import ray
from rationai.tiling import tiling
from rationai.tiling.modules.masks import PyvipsMask
from rationai.tiling.modules.tile_sources import OpenSlideTileSource
from rationai.tiling.modules.tiling_module import proxy
from rationai.tiling.typing import Sized2, TiledSlideMetadata, TileMetadata
from rationai.tiling.writers import save_mlflow_dataset
from sklearn.model_selection import train_test_split

from preprocessing.paths import DATAFRAME_PATH, SLIDES_PATH, TISSUE_MASKS_PATH
from preprocessing.tissue_regions import add_regions
from preprocessing.typing import (
    UlcerativeColitisSlideMetadata,
    UlcerativeColitisTileMetadata,
)


TILE_SIZE = 224
STRIDE = 112


class TissueMask(PyvipsMask[UlcerativeColitisTileMetadata]):
    def forward_tile(
        self,
        tile_labels: UlcerativeColitisTileMetadata,
        class_overlaps: dict[int, float],
    ) -> UlcerativeColitisTileMetadata | None:
        if class_overlaps.get(0, 0) > 0.5:
            return None
        return tile_labels

    def forward(
        self,
        mask_path: Path | str,
        slide_extent: Sized2[int],
        tiles_labels: Iterable[TileMetadata],
    ) -> list[UlcerativeColitisTileMetadata]:
        tiles = super().forward(
            mask_path=mask_path,
            slide_extent=slide_extent,
            tiles_labels=tiles_labels,
        )

        return add_regions(tiles, tile_size=TILE_SIZE, stride=STRIDE)

    @proxy(forward)
    def __call__(self) -> None: ...


source = OpenSlideTileSource(mpp=0.5, tile_extent=TILE_SIZE, stride=STRIDE)
tissue_mask = TissueMask(
    tile_extent=source.tile_extent, absolute_roi_extent=112, relative_roi_offset=0
)
df = pd.read_csv(DATAFRAME_PATH, index_col=0).query("Lokalita != 'ileum'")


def stem_to_case_id(stem: str) -> str:
    case_id, year, _ = stem.split("_", maxsplit=2)
    return f"{case_id.zfill(5)}/{year}"


def get_slides(df: pd.DataFrame) -> list[Path]:
    return [
        slide_path
        for slide_path in SLIDES_PATH.glob("*.tiff")
        if stem_to_case_id(slide_path.stem) in df.index
    ]


def get_metadata(slide_path: Path, df_metadata: pd.DataFrame) -> dict[str, Any]:
    index = stem_to_case_id(slide_path.stem)
    return {
        "nancy_index": int(pd.to_numeric(df_metadata.loc[index, "Nancy"])),
        "location": df_metadata.loc[index, "Lokalita"],
        "diagnosis": df_metadata.loc[index, "Diagnoza"],
    }


@ray.remote
def handler(slide_path: Path) -> TiledSlideMetadata:
    slide, tiles = source(slide_path)

    tissue_mask_path = TISSUE_MASKS_PATH / slide_path.name

    tiles = tissue_mask(tissue_mask_path, slide.extent, tiles)
    slide = UlcerativeColitisSlideMetadata(
        **asdict(slide), **get_metadata(slide_path, df)
    )

    return slide, tiles


def main() -> None:
    # 70 / 15 / 15 - train / test preliminary / test final
    train, test = train_test_split(
        df, test_size=0.3, stratify=df["Nancy"], random_state=42
    )

    test_preliminary, test_final = train_test_split(
        test, test_size=0.5, stratify=test["Nancy"], random_state=42
    )

    train_slides_df, train_tiles_df = tiling(slides=get_slides(train), handler=handler)
    test_preliminary_slides_df, test_preliminary_tiles_df = tiling(
        slides=get_slides(test_preliminary), handler=handler
    )
    test_final_slides_df, test_final_tiles_df = tiling(
        slides=get_slides(test_final), handler=handler
    )

    mlflow.set_experiment(experiment_name="Ulcerative Colitis")
    with mlflow.start_run(run_name="📂 Dataset: Ulcerative Colitis"):
        save_mlflow_dataset(
            slides=train_slides_df,
            tiles=train_tiles_df,
            dataset_name="Ulcerative Colitis - train",
        )
        save_mlflow_dataset(
            slides=test_preliminary_slides_df,
            tiles=test_preliminary_tiles_df,
            dataset_name="Ulcerative Colitis - test preliminary",
        )
        save_mlflow_dataset(
            slides=test_final_slides_df,
            tiles=test_final_tiles_df,
            dataset_name="Ulcerative Colitis - test final",
        )


if __name__ == "__main__":
    main()
