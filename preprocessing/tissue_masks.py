from pathlib import Path
from tempfile import TemporaryDirectory
from typing import cast

import hydra
import mlflow.artifacts
import pandas as pd
import pyvips
import ray
from omegaconf import DictConfig
from openslide import OpenSlide
from rationai.masks import (
    process_items,
    slide_resolution,
    tissue_mask,
    write_big_tiff,
)
from rationai.mlkit import autolog, with_cli_args
from rationai.mlkit.lightning.loggers import MLFlowLogger


ray.init(runtime_env={"excludes": [".git", ".venv"]})


@ray.remote(memory=4 * 1024**3)
def process_slide(slide_path: str, level: int, output_path: Path) -> None:
    with OpenSlide(slide_path) as slide:
        mpp_x, mpp_y = slide_resolution(slide, level)

    slide = cast("pyvips.Image", pyvips.Image.new_from_file(slide_path, level=level))
    mask = tissue_mask(slide, mpp=(mpp_x + mpp_y) / 2)
    mask_path = output_path / Path(slide_path).with_suffix(".tiff").name

    write_big_tiff(mask, path=mask_path, mpp_x=mpp_x, mpp_y=mpp_y)


def download_dataframe(uri: str) -> pd.DataFrame:
    path = mlflow.artifacts.download_artifacts(artifact_uri=uri)
    df = pd.read_csv(path)
    return df


@with_cli_args(["+preprocessing=tissue_masks"])
@hydra.main(config_path="../configs", config_name="preprocessing", version_base=None)
@autolog
def main(config: DictConfig, logger: MLFlowLogger) -> None:
    df = download_dataframe(config.dataset.uri)

    with TemporaryDirectory() as output_dir:
        process_items(
            df["path"].to_list(),
            process_item=process_slide,
            fn_kwargs={
                "level": config.level,
                "output_path": Path(output_dir),
            },
            max_concurrent=config.max_concurrent,
        )

        logger.log_artifacts(local_dir=output_dir, artifact_path=config.artifact_path)


if __name__ == "__main__":
    main()
