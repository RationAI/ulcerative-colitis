from pathlib import Path
from typing import Any, TypedDict, cast

import hydra
import mlflow.artifacts
import pandas as pd
import ray
from omegaconf import DictConfig
from rationai.mlkit import with_cli_args
from rationai.mlkit.autolog import autolog
from rationai.mlkit.lightning.loggers import MLFlowLogger
from rationai.tiling.writers import save_mlflow_dataset
from ratiopath.ray import read_slides
from ratiopath.tiling import grid_tiles, tile_overlay_overlap
from ratiopath.tiling.utils import row_hash
from ray.data.expressions import col
from shapely import Polygon
from shapely.geometry import box


QC_BLUR_MEAN_COLUMN = "mean_coverage(Piqe)"
QC_ARTIFACTS_MEAN_COLUMN = "mean_coverage(ResidualArtifactsAndCoverage)"
QC_SUBFOLDERS = {"blur": "blur_per_pixel", "artifacts": "artifacts_per_pixel"}


class _RayCpuResources(TypedDict):
    num_cpus: float


class _RayMemResources(TypedDict):
    memory: int


LO_CPU: _RayCpuResources = {"num_cpus": 0.1}
HI_CPU: _RayCpuResources = {"num_cpus": 0.2}
LO_MEM: _RayMemResources = {"memory": 128 * 1024**2}
HI_MEM: _RayMemResources = {"memory": 256 * 1024**2}


def add_nancy_index(row: dict[str, Any], df: pd.DataFrame) -> dict[str, Any]:
    row["nancy_index"] = df.loc[Path(row["path"]).stem, "nancy"]
    return row


def qc_agg(row: dict[str, Any], df: pd.DataFrame) -> dict[str, Any]:
    qc_df = cast("pd.Series", df.loc[Path(row["path"]).stem])

    row["blur_mean"] = qc_df[QC_BLUR_MEAN_COLUMN]
    row["artifacts_mean"] = qc_df[QC_ARTIFACTS_MEAN_COLUMN]

    return row


def add_fold(row: dict[str, Any], df: pd.DataFrame) -> dict[str, Any]:
    row["fold"] = df.loc[Path(row["path"]).stem, "fold"]
    return row


def add_mask_paths(
    row: dict[str, Any], qc_folder: Path, tissue_folder: Path
) -> dict[str, Any]:
    stem = Path(row["path"]).stem
    row["tissue_mask_path"] = str(tissue_folder / f"{stem}.tiff")
    for key, subfolder in QC_SUBFOLDERS.items():
        row[f"{key}_mask_path"] = str(qc_folder / subfolder / f"{stem}.tiff")
    return row


def create_tissue_roi(tile_extent: int) -> Polygon:
    offset = tile_extent // 4
    size = tile_extent // 2
    return box(offset, offset, offset + size, offset + size)


def create_qc_roi(tile_extent: int) -> Polygon:
    return box(0, 0, tile_extent, tile_extent)


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
            "tissue_mask_path": row["tissue_mask_path"],
            "blur_mask_path": row["blur_mask_path"],
            "artifacts_mask_path": row["artifacts_mask_path"],
        }
        for x, y in grid_tiles(
            slide_extent=(row["extent_x"], row["extent_y"]),
            tile_extent=(row["tile_extent_x"], row["tile_extent_y"]),
            stride=(row["stride_x"], row["stride_y"]),
        )
    ]


def extract_coverages(row: dict[str, Any], *cols) -> dict[str, Any]:
    for c in cols:
        overlap = row[f"{c}_overlap"]
        zero_overlap = overlap.get("0", 0)
        if zero_overlap is None:
            row[c] = 1.0
        else:
            row[c] = 1.0 - zero_overlap
    return row


def filter_tissue(row: dict[str, Any], threshold: float) -> bool:
    return row["tissue"] >= threshold


def select(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "slide_id": row["slide_id"],
        "x": row["tile_x"],
        "y": row["tile_y"],
        "tissue": row["tissue"],
        "blur": row["blur"],
        "artifacts": row["artifacts"],
    }


def tiling(
    df: pd.DataFrame,
    qc_folder: Path,
    tissue_folder: Path,
    tile_extent: int,
    stride: int,
    mpp: float,
    tissue_threshold: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    qc_df = pd.read_csv(qc_folder / "qc_metrics.csv", index_col="slide_name")
    paths = df["path"].tolist()

    slides = (
        read_slides(paths, tile_extent=tile_extent, stride=stride, mpp=mpp)
        .map(row_hash, **LO_CPU, **LO_MEM)
        .map(add_nancy_index, fn_args=(df,), **LO_CPU, **LO_MEM)  # type: ignore[reportArgumentType]
        .map(qc_agg, fn_args=(qc_df,), **HI_CPU, **LO_MEM)  # type: ignore[reportArgumentType]
    )

    if "fold" in df.columns:
        slides = slides.map(add_fold, fn_args=(df,), **LO_CPU, **LO_MEM)  # type: ignore[reportArgumentType]

    tissue_roi = create_tissue_roi(tile_extent)
    qc_roi = create_qc_roi(tile_extent)

    tiles = (
        slides.map(
            add_mask_paths,  # type: ignore[reportArgumentType]
            fn_args=(qc_folder, tissue_folder),
            **LO_CPU,
            **LO_MEM,
        )
        .flat_map(tile, **HI_CPU, **LO_MEM)
        .repartition(target_num_rows_per_block=4096)
        .with_column(
            "tissue_overlap",
            tile_overlay_overlap(
                tissue_roi,
                col("tissue_mask_path"),
                col("tile_x"),
                col("tile_y"),
                col("mpp_x"),
                col("mpp_y"),
            ),  # type: ignore[reportCallIssue]
            **HI_CPU,
            **HI_MEM,
        )
        .map(extract_coverages, fn_args=("tissue",), **LO_CPU, **LO_MEM)  # type: ignore[reportArgumentType]
        .filter(filter_tissue, fn_args=(tissue_threshold,), **LO_CPU, **LO_MEM)  # type: ignore[reportArgumentType]
        .with_column(
            "blur_overlap",
            tile_overlay_overlap(
                qc_roi,
                col("blur_mask_path"),
                col("tile_x"),
                col("tile_y"),
                col("mpp_x"),
                col("mpp_y"),
            ),  # type: ignore[reportCallIssue]
            **HI_CPU,
            **HI_MEM,
        )
        .with_column(
            "artifacts_overlap",
            tile_overlay_overlap(
                qc_roi,
                col("artifacts_mask_path"),
                col("tile_x"),
                col("tile_y"),
                col("mpp_x"),
                col("mpp_y"),
            ),  # type: ignore[reportCallIssue]
            **HI_CPU,
            **HI_MEM,
        )
        .map(extract_coverages, fn_args=("blur", "artifacts"), **LO_CPU, **LO_MEM)  # type: ignore[reportArgumentType]
        .map(select, **LO_CPU, **LO_MEM)
    )

    return slides.to_pandas(), tiles.to_pandas()


@with_cli_args(["+preprocessing=tiling"])
@hydra.main(config_path="../configs", config_name="preprocessing", version_base=None)
@autolog
def main(config: DictConfig, logger: MLFlowLogger) -> None:
    qc_folder = Path(
        mlflow.artifacts.download_artifacts(config.dataset.mlflow_uris.qc_mask)
    )
    tissue_folder = Path(
        mlflow.artifacts.download_artifacts(config.dataset.mlflow_uris.tissue_mask)
    )

    for name, split_uri in config.dataset.mlflow_uris.splits.items():
        split = pd.read_csv(mlflow.artifacts.download_artifacts(split_uri))
        df_slides, df_tiles = tiling(
            split,
            qc_folder=qc_folder,
            tissue_folder=tissue_folder,
            tile_extent=config.tile_extent,
            stride=config.stride,
            mpp=config.mpp,
            tissue_threshold=config.tissue_threshold,
        )
        save_mlflow_dataset(
            df_slides, df_tiles, f"{name} - {config.dataset.institution}"
        )


if __name__ == "__main__":
    ray.init(runtime_env={"excludes": [".git", ".venv"]})
    main()
