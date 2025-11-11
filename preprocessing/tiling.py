from functools import partial
from pathlib import Path
from typing import Any, cast

import hydra
import mlflow.artifacts
import numpy as np
import pandas as pd
import ray
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig
from rationai.mlkit.autolog import autolog
from rationai.tiling.writers import save_mlflow_dataset
from ratiopath.ray import read_slides
from ratiopath.tiling import grid_tiles
from ratiopath.tiling.utils import row_hash
from sklearn.model_selection import train_test_split


ray.init(runtime_env={"excludes": [".git", ".venv"]})


def stem_to_case_id(stem: str) -> str:
    case_id, year, *_ = stem.split("_")
    return f"{case_id}/{year}"


def get_slides(df: pd.DataFrame, slides_folder: Path) -> list[str]:
    return [
        str(slide_path)
        for slide_path in slides_folder.glob("*.tiff")
        if stem_to_case_id(slide_path.stem) in df.index
    ]


def get_qc_folder(config: DictConfig) -> Path:
    if config.qc_download:
        return Path(mlflow.artifacts.download_artifacts(config.qc_uri))
    return Path(config.qc_folder)


def get_tissue_folder(config: DictConfig) -> Path:
    if config.tissue_download:
        return Path(mlflow.artifacts.download_artifacts(config.tissue_uri))
    return Path(config.tissue_folder)


def nancy(row: dict[str, Any], df: pd.DataFrame) -> dict[str, Any]:
    case_id = stem_to_case_id(Path(row["path"]).stem)
    row["nancy_index"] = df.loc[case_id, "Nancy"]
    return row


def qc_agg(row: dict[str, Any], qc_folder: Path) -> dict[str, Any]:
    qc_file = qc_folder / f"{Path(row['path']).stem}.csv"
    qc_df = cast("pd.Series", pd.read_csv(qc_file).T.squeeze())

    row["blur_mean"] = qc_df["mean_coverage(Piqe)"]
    row["artifacts_mean"] = qc_df["mean_coverage(ResidualArtifactsAndCoverage)"]

    return row


def tile(row: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "tile_x": x,
            "tile_y": y,
            "path": row["path"],
            "slide_id": row["id"],
            "level": row["level"],
            "tile_extent_x": row["tile_extent_x"],
            "tile_extent_y": row["tile_extent_y"],
            "mpp_x": row["mpp_x"],
            "mpp_y": row["mpp_y"],
        }
        for x, y in grid_tiles(
            slide_extent=(row["extent_x"], row["extent_y"]),
            tile_extent=(row["tile_extent_x"], row["tile_extent_y"]),
            stride=(row["stride_x"], row["stride_y"]),
        )
    ]


def read_tile_overlays(overlay_path: str, df: pd.DataFrame) -> pd.Series:
    from ratiopath.openslide import OpenSlide

    with OpenSlide(overlay_path) as overlay:

        def get_tile(row: pd.Series) -> np.ndarray:
            resolution = (row["mpp_x"] + row["mpp_y"]) / 2
            level = overlay.closest_level(resolution)
            overlay_resolution = overlay.slide_resolution(level)
            resolution_factor = np.asarray(overlay_resolution) / np.asarray(resolution)

            roi_coords = tuple(
                np.round(
                    np.asarray((row["tile_x"], row["tile_y"])) / resolution_factor
                ).astype(int)
            )
            roi_extent = tuple(
                np.round(
                    np.asarray((row["tile_extent_x"], row["tile_extent_y"]))
                    / resolution_factor
                ).astype(int)
            )

            rgba_region = overlay.read_region_relative(roi_coords, level, roi_extent)
            grayscale_region = rgba_region.convert("L")
            return np.asarray(grayscale_region)

        return df.apply(get_tile, axis=1)


def tissue(batch: dict[str, Any], tissue_folder: Path) -> dict[str, Any]:
    df = pd.DataFrame(batch)
    for path, group in df.groupby("path"):
        tissue_file = tissue_folder / f"{Path(str(path)).stem}.tiff"
        tile_series = read_tile_overlays(str(tissue_file), group)

        df.loc[group.index, "tissue"] = tile_series.apply(
            lambda x: (x == 255).sum() / x.size
        )

    batch["tissue"] = df["tissue"].tolist()
    return batch


def filter_tissue(row: dict[str, Any], threshold: float) -> bool:
    return row["tissue"] >= threshold


def qc(batch: dict[str, Any], qc_folder: Path) -> dict[str, Any]:
    df = pd.DataFrame(batch)
    for path, group in df.groupby("path"):
        for qc_prefix in [
            "Piqe_piqe_median_activity_mask",
            "ResidualArtifactsAndCoverage_coverage_mask",
        ]:
            qc_file = qc_folder / f"{qc_prefix}_{Path(str(path)).stem}.tiff"
            tile_series = read_tile_overlays(str(qc_file), group)

            df.loc[group.index, qc_prefix] = tile_series.apply(lambda x: x.mean() / 255)

    batch["blur"] = df["Piqe_piqe_median_activity_mask"].tolist()
    batch["artifacts"] = df["ResidualArtifactsAndCoverage_coverage_mask"].tolist()
    return batch


def select(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "slide_id": row["slide_id"],
        "x": row["tile_x"],
        "y": row["tile_y"],
        "tissue": row["tissue"],
        "blur": row["blur"],
        "artifacts": row["artifacts"],
    }


def tiling(df: pd.DataFrame, config: DictConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    qc_folder = get_qc_folder(config)
    tissue_folder = get_tissue_folder(config)

    slides = read_slides(
        get_slides(df, Path(config.slides_folder)),
        tile_extent=config.tile_extent,
        stride=config.stride,
        mpp=config.mpp,
    )
    slides = slides.map(row_hash, num_cpus=0.1, memory=128 * 1024**2)
    slides = slides.map(partial(nancy, df=df), num_cpus=0.1, memory=128 * 1024**2)
    slides = slides.map(
        partial(qc_agg, qc_folder=qc_folder), num_cpus=0.2, memory=128 * 1024**2
    )

    tiles = slides.flat_map(tile, num_cpus=0.2, memory=128 * 1024**2)
    tiles = tiles.repartition(target_num_rows_per_block=4096)
    tiles = tiles.map_batches(
        partial(tissue, tissue_folder=tissue_folder), num_cpus=0.2, memory=1024**3
    )
    tiles = tiles.filter(
        partial(filter_tissue, threshold=config.tissue_threshold),
        num_cpus=0.1,
        memory=128 * 1024**2,
    )
    tiles = tiles.map_batches(
        partial(qc, qc_folder=qc_folder), num_cpus=0.2, memory=1024**3
    )
    tiles = tiles.map(select, num_cpus=0.1, memory=128 * 1024**2)

    return slides.to_pandas(), tiles.to_pandas()


@hydra.main(config_path="../configs", config_name="tiling", version_base=None)
@autolog
def main(config: DictConfig, logger: Logger | None = None) -> None:
    df = pd.read_csv(config.table_path, index_col=0)
    if config.cohort == "ikem":
        df = df.query("Lokalita != 'ileum'")

    train: pd.DataFrame
    test: pd.DataFrame
    test_preliminary: pd.DataFrame
    test_final: pd.DataFrame

    if config.splits.train == 0.0:
        train = pd.DataFrame(columns=df.columns)
        test = df
    else:
        train, test = train_test_split(
            df, train_size=config.splits.train, stratify=df["Nancy"], random_state=42
        )

    preliminary_size = (1.0 - config.splits.train) * config.splits.preliminary
    test_preliminary, test_final = train_test_split(
        test, test_size=preliminary_size, stratify=test["Nancy"], random_state=42
    )

    if not train.empty:
        train_slides, train_tiles = tiling(train, config)
        save_mlflow_dataset(train_slides, train_tiles, f"train - {config.cohort}")

    if not test_preliminary.empty:
        test_preliminary_slides, test_preliminary_tiles = tiling(
            test_preliminary, config
        )
        save_mlflow_dataset(
            test_preliminary_slides,
            test_preliminary_tiles,
            f"test preliminary - {config.cohort}",
        )

    if not test_final.empty:
        test_final_slides, test_final_tiles = tiling(test_final, config)
        save_mlflow_dataset(
            test_final_slides, test_final_tiles, f"test final - {config.cohort}"
        )


if __name__ == "__main__":
    main()
