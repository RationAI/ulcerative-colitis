from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import mlflow
import pandas as pd
import ray
from rationai.tiling import tiling
from rationai.tiling.modules.masks import PyvipsMask
from rationai.tiling.modules.tile_sources import OpenSlideTileSource
from rationai.tiling.typing import TiledSlideMetadata, TileMetadata
from rationai.tiling.writers import save_mlflow_dataset
from sklearn.model_selection import train_test_split


BASE_FOLDER = Path("/mnt/data/Projects/inflammatory_bowel_dissease/ulcerative_colitis/")
SLIDES_PATH = BASE_FOLDER / "data_tiff" / "20x"
DATAFRAME_PATH = BASE_FOLDER / "data_czi" / "Fab_IBD_AI_12_2024.csv"
TISSUE_MASKS_PATH = BASE_FOLDER / "tissue_masks" / "20x"


@dataclass
class UlcerativeColitisTileMetadata(TileMetadata):
    nancy_index: int
    location: str
    diagnosis: str


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


def stem_to_case_id(stem: str) -> str:
    case_id, year, _ = stem.split("_", maxsplit=2)
    return f"{case_id.zfill(5)}/{year}"


def train_test_split_cases(
    slides: list[Path], test_size: float, random_state: int = 42
) -> tuple[list[Path], list[Path]]:
    cases = set()
    for slide in slides:
        cases.add(stem_to_case_id(slide.stem))

    train_cases, test_cases = train_test_split(
        list(cases), test_size=test_size, random_state=random_state
    )

    return (
        [slide for slide in slides if stem_to_case_id(slide.stem) in train_cases],
        [slide for slide in slides if stem_to_case_id(slide.stem) in test_cases],
    )


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
    tiles = [
        UlcerativeColitisTileMetadata(**asdict(tile), **get_metadata(slide_path, df))
        for tile in tiles
    ]

    return slide, tiles


def main() -> None:
    all_slides = [
        slide_path
        for slide_path in SLIDES_PATH.rglob("*.tiff")
        if stem_to_case_id(slide_path.stem) in df.index
    ]

    # 70 / 10 / 10 / 10 - train / val / test1 / test2
    train_val_slides, test_slides = train_test_split_cases(all_slides, test_size=0.2)
    train_slides, val_slides = train_test_split_cases(train_val_slides, test_size=0.125)
    test1_slides, test2_slides = train_test_split_cases(test_slides, test_size=0.5)

    train_slides_df, train_tiles_df = tiling(slides=train_slides, handler=handler)
    val_slides_df, val_tiles_df = tiling(slides=val_slides, handler=handler)
    test1_slides_df, test1_tiles_df = tiling(slides=test1_slides, handler=handler)
    test2_slides_df, test2_tiles_df = tiling(slides=test2_slides, handler=handler)

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
            slides=test1_slides_df,
            tiles=test1_tiles_df,
            dataset_name="Ulcerative Colitis - test1",
        )
        save_mlflow_dataset(
            slides=test2_slides_df,
            tiles=test2_tiles_df,
            dataset_name="Ulcerative Colitis - test2",
        )


if __name__ == "__main__":
    main()
