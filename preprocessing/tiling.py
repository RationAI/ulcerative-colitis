from math import isclose
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
from ratiopath.tiling import grid_tiles, relative_tile_overlay, tile_overlay
from ratiopath.tiling.utils import row_hash
from sklearn.model_selection import train_test_split


ray.init(runtime_env={"excludes": [".git", ".venv"]})


QC_SUBFOLDERS = {"blur": "blur_per_pixel", "artifacts": "artifacts_per_pixel"}


class _RayCpuResources(TypedDict):
    num_cpus: float


class _RayMemResources(TypedDict):
    memory: int


LO_CPU: _RayCpuResources = {"num_cpus": 0.1}
HI_CPU: _RayCpuResources = {"num_cpus": 0.2}
LO_MEM: _RayMemResources = {"memory": 128 * 1024**2}
HI_MEM: _RayMemResources = {"memory": 1024**3}


def download_dataset(uri: str) -> pd.DataFrame:
    path = mlflow.artifacts.download_artifacts(artifact_uri=uri)
    df = pd.read_csv(path)
    return df


def split_dataset(
    dataset: pd.DataFrame, splits: dict[str, float], random_state: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    assert isclose(
        splits["train"] + splits["test_preliminary"] + splits["test_final"], 1.0
    ), "Splits must sum to 1.0"

    train: pd.DataFrame
    test: pd.DataFrame
    test_preliminary: pd.DataFrame
    test_final: pd.DataFrame

    if splits["train"] == 0.0:
        train = pd.DataFrame(columns=dataset.columns)
        test = dataset
    else:
        train, test = train_test_split(
            dataset,
            train_size=splits["train"],
            stratify=dataset["nancy"],
            random_state=random_state,
        )

    if splits["test_preliminary"] == 0.0:
        test_preliminary = pd.DataFrame(columns=test.columns)
        test_final = test
    else:
        preliminary_size = splits["test_preliminary"] / (1.0 - splits["train"])
        test_preliminary, test_final = train_test_split(
            test,
            train_size=preliminary_size,
            stratify=test["nancy"],
            random_state=random_state,
        )

    return train, test_preliminary, test_final


def nancy(row: dict[str, Any], df: pd.DataFrame) -> dict[str, Any]:
    row["nancy_index"] = df.loc[Path(row["path"]).stem, "nancy"]
    return row


def qc_agg(row: dict[str, Any], df: pd.DataFrame) -> dict[str, Any]:
    qc_df = cast("pd.Series", df.loc[Path(row["path"]).stem])

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


def tissue(row: dict[str, Any], tissue_folder: Path) -> dict[str, Any]:
    tissue_file = tissue_folder / Path(row["path"]).with_suffix(".tiff").name
    overlay = relative_tile_overlay(
        tissue_file,
        (row["mpp_x"], row["mpp_y"]),
        (row["tile_x"], row["tile_y"]),
        (row["tile_extent_x"] // 4, row["tile_extent_y"] // 4),
        (row["tile_extent_x"] // 2, row["tile_extent_y"] // 2),
    )

    row["tissue"] = (overlay == 255).sum() / overlay.size
    return row


def filter_tissue(row: dict[str, Any], threshold: float) -> bool:
    return row["tissue"] >= threshold


def qc(row: dict[str, Any], qc_folder: Path) -> dict[str, Any]:
    for qc_key, subfolder in QC_SUBFOLDERS.items():
        qc_file = qc_folder / subfolder / Path(row["path"]).with_suffix(".tiff").name
        overlay = tile_overlay(
            qc_file,
            (row["mpp_x"], row["mpp_y"]),
            (row["tile_x"], row["tile_y"]),
            (row["tile_extent_x"], row["tile_extent_y"]),
        )

        row[qc_key] = overlay.mean() / 255.0

    return row


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
    qc_mask_uri: str,
    tissue_mask_uri: str,
    tile_extent: int,
    stride: int,
    mpp: float,
    tissue_threshold: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    qc_folder = Path(mlflow.artifacts.download_artifacts(qc_mask_uri))
    qc_df = pd.read_csv(qc_folder / "qc_metrics.csv", index_col="slide_name")
    tissue_folder = Path(mlflow.artifacts.download_artifacts(tissue_mask_uri))
    paths = df["slide_path"].tolist()

    slides = (
        read_slides(paths, tile_extent=tile_extent, stride=stride, mpp=mpp)
        .map(row_hash, **LO_CPU, **LO_MEM)
        .map(nancy, fn_args=(df,), **LO_CPU, **LO_MEM)  # type: ignore[reportArgumentType]
        .map(qc_agg, fn_args=(qc_df,), **HI_CPU, **LO_MEM)  # type: ignore[reportArgumentType]
    )

    tiles = (
        slides.flat_map(tile, **HI_CPU, **LO_MEM)
        .repartition(target_num_rows_per_block=4096)
        .map(tissue, fn_args=(tissue_folder,), **HI_CPU, **HI_MEM)  # type: ignore[reportArgumentType]
        .filter(filter_tissue, fn_args=(tissue_threshold,), **LO_CPU, **LO_MEM)  # type: ignore[reportArgumentType]
        .map(qc, fn_args=(qc_folder,), **HI_CPU, **HI_MEM)  # type: ignore[reportArgumentType]
        .map(select, **LO_CPU, **LO_MEM)
    )

    return slides.to_pandas(), tiles.to_pandas()


@with_cli_args(["+preprocessing=tiling"])
@hydra.main(config_path="../configs", config_name="preprocessing", version_base=None)
@autolog
def main(config: DictConfig, logger: MLFlowLogger) -> None:
    dataset = download_dataset(config.dataset.uri)

    train, test_preliminary, test_final = split_dataset(dataset, config.splits)

    for df, name in [
        (train, "train"),
        (test_preliminary, "test preliminary"),
        (test_final, "test final"),
    ]:
        if df.empty:
            continue

        df_slides, df_tiles = tiling(
            df,
            qc_mask_uri=config.qc_mask.uri,
            tissue_mask_uri=config.tissue_mask.uri,
            tile_extent=config.tile_extent,
            stride=config.stride,
            mpp=config.mpp,
            tissue_threshold=config.tissue_threshold,
        )
        save_mlflow_dataset(
            df_slides, df_tiles, f"{name} - {config.dataset.institution}"
        )


if __name__ == "__main__":
    main()
