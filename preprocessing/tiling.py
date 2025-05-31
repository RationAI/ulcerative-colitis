from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import mlflow
import pandas as pd
import ray
from rationai.tiling import tiling
from rationai.tiling.modules.masks import PyvipsMask
from rationai.tiling.modules.tile_sources import OpenSlideTileSource
from rationai.tiling.modules.tile_sources.openslide_tile_source import OpenSlideMetadata
from rationai.tiling.typing import TiledSlideMetadata, TileMetadata
from rationai.tiling.writers import save_mlflow_dataset
from sklearn.model_selection import train_test_split
from tqdm import tqdm


BASE_FOLDER = Path("/mnt/data/Projects/inflammatory_bowel_dissease/ulcerative_colitis/")
SLIDES_PATH = BASE_FOLDER / "data_tiff" / "20x"
DATAFRAME_PATH = BASE_FOLDER / "data_czi" / "IBD_AI.csv"
TISSUE_MASKS_PATH = BASE_FOLDER / "tissue_masks" / "20x"

TILE = 224
STRIDE = 112


@dataclass
class UlcerativeColitisSlideMetadata(OpenSlideMetadata):
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


source = OpenSlideTileSource(mpp=0.5, tile_extent=TILE, stride=STRIDE)
tissue_mask = TissueMask(
    tile_extent=source.tile_extent, absolute_roi_extent=112, relative_roi_offset=0
)
df = pd.read_csv(DATAFRAME_PATH, index_col=0).query("Lokalita != 'ileum'")


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


def dfs_neighbors(tiles_df: pd.DataFrame, idx: int) -> Iterable[int]:
    current_tile = tiles_df.loc[idx]

    for dx in range(-TILE, TILE + 1, STRIDE):
        for dy in range(-TILE, TILE + 1, STRIDE):
            if dx == 0 and dy == 0:
                continue

            neighbor = tiles_df[
                (tiles_df["x"] == current_tile["x"] + dx)
                & (tiles_df["y"] == current_tile["y"] + dy)
                & (tiles_df["tissue_region"] == -1)
            ]

            if not neighbor.empty:
                yield neighbor.index[0]


def dfs(tiles_df: pd.DataFrame, idx: int, label: int) -> pd.DataFrame:
    stack = [idx]
    tiles_df.at[idx, "tissue_region"] = label

    while stack:
        current_idx = stack.pop()
        for neighbor_idx in dfs_neighbors(tiles_df, current_idx):
            tiles_df.at[neighbor_idx, "tissue_region"] = label
            stack.append(neighbor_idx)

    return tiles_df


def add_regions(slides_df: pd.DataFrame, tiles_df: pd.DataFrame) -> pd.DataFrame:
    tiles_df["tissue_region"] = -1

    for slide_id in tqdm(slides_df["id"]):
        slide_tiles = tiles_df.query(f"slide_id == {slide_id}")
        label = 0

        for idx in slide_tiles.index:
            if slide_tiles.at[idx, "tissue_region"] == -1:
                slide_tiles = dfs(slide_tiles, int(idx), label)
                label += 1

        tiles_df.update(slide_tiles)

    return tiles_df


def main() -> None:
    # 70 / 15 / 15 - train / test preliminary / test final
    train, _test = train_test_split(
        df, test_size=0.3, stratify=df["Nancy"], random_state=42
    )

    test_preliminary, test_final = train_test_split(
        _test, test_size=0.5, stratify=_test["Nancy"], random_state=42
    )

    train_slides_df, train_tiles_df = tiling(slides=get_slides(train), handler=handler)
    test_preliminary_slides_df, test_preliminary_tiles_df = tiling(
        slides=get_slides(test_preliminary), handler=handler
    )
    test_final_slides_df, test_final_tiles_df = tiling(
        slides=get_slides(test_final), handler=handler
    )

    mlflow.set_experiment(experiment_name="IKEM")
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
