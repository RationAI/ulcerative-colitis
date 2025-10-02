from pathlib import Path
from typing import cast

import hydra
import mlflow.artifacts
import numpy as np
import pandas as pd
import pyvips
import ray
from lightning.pytorch.loggers import Logger
from mlflow import MlflowClient
from omegaconf import DictConfig
from rationai.masks import process_items, tile_mask, write_big_tiff
from rationai.mlkit import autolog
from rationai.mlkit.lightning.loggers import MLFlowLogger
from ray._private.worker import RemoteFunction0


ray.init(runtime_env={"excludes": [".git", ".venv"]})


def download_slide_tiles(uris: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    slidess, tiless = [], []
    for uri in uris:
        path = Path(mlflow.artifacts.download_artifacts(uri))
        slidess.append(pd.read_parquet(path / "slides.parquet"))
        tiless.append(pd.read_parquet(path / "tiles.parquet"))

    slides = pd.concat(slidess).reset_index(drop=True)
    tiles = pd.concat(tiless).reset_index(drop=True)
    return slides, tiles


def process_slide(
    slide: pd.Series,
    output_path: Path,
    tiles_ref: ray.ObjectRef,
) -> None:
    tiles = cast("pd.DataFrame", ray.get(tiles_ref))
    slide_tiles = tiles[tiles["slide_id"] == slide.id]

    mask = tile_mask(
        slide_tiles,
        tile_extent=(slide["tile_extent_x"], slide["tile_extent_y"]),
        size=(slide["extent_x"], slide["extent_y"]),
    )

    mask_path = output_path / "outlines" / f"{Path(slide['path']).stem}.tiff"
    write_big_tiff(
        pyvips.Image.new_from_array(np.array(mask)),
        mask_path,
        mpp_x=slide["mpp_x"],
        mpp_y=slide["mpp_y"],
    )


def make_remote_process_slide(
    output_path: Path,
    tiles_ref: ray.ObjectRef,
) -> RemoteFunction0[None, pd.Series]:
    @ray.remote
    def _process_slide(slide_meta: pd.Series) -> None:
        process_slide(slide_meta, output_path, tiles_ref)

    return _process_slide


@hydra.main(config_path="../../configs", config_name="tile_masks", version_base=None)
@autolog
def main(config: DictConfig, logger: Logger | None = None) -> None:
    assert logger is not None, "Need logger"
    assert isinstance(logger, MLFlowLogger), "Need MLFlowLogger"
    assert isinstance(logger.experiment, MlflowClient), "Need MlflowClient"
    assert logger.run_id is not None, "Need run_id"

    slides, tiles = download_slide_tiles(config.tiling_uris)
    tiles_ref = ray.put(tiles)

    output_path = Path(config.output_path)

    process_items(
        (slide for _, slide in slides.iterrows()),
        make_remote_process_slide(output_path, tiles_ref),
    )

    logger.experiment.log_artifacts(
        run_id=logger.run_id, local_dir=str(output_path), artifact_path="tile_masks"
    )


if __name__ == "__main__":
    main()
