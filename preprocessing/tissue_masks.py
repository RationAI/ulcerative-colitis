from pathlib import Path
from typing import cast

import hydra
import pandas as pd
import pyvips
import ray
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig
from openslide import OpenSlide
from rationai.masks import (
    process_items,
    slide_resolution,
    tissue_mask,
    write_big_tiff,
)
from rationai.mlkit.autolog import autolog
from rationai.mlkit.lightning.loggers import MLFlowLogger
from ray._private.worker import RemoteFunction0

from preprocessing.slides import get_slides


ray.init(runtime_env={"excludes": [".git", ".venv"]})


def process_slide(slide_path: Path, level: int, output_path: Path) -> None:
    with OpenSlide(slide_path) as slide:
        mpp_x, mpp_y = slide_resolution(slide, level=level)

    slide = cast("pyvips.Image", pyvips.Image.new_from_file(slide_path, level=level))
    mask = tissue_mask(slide, mpp=(mpp_x + mpp_y) / 2)
    mask_path = output_path / slide_path.with_suffix(".tiff").name

    write_big_tiff(mask, path=mask_path, mpp_x=mpp_x, mpp_y=mpp_y)


def make_remote_process_slide(
    level: int, output_path: Path
) -> RemoteFunction0[None, Path]:
    @ray.remote
    def _process_slide(slide_path: Path) -> None:
        return process_slide(slide_path, level, output_path)

    return _process_slide


@hydra.main(config_path="../configs", config_name="tissue_masks", version_base=None)
@autolog
def main(config: DictConfig, logger: Logger | None = None) -> None:
    assert logger is not None, "Need logger"
    assert isinstance(logger, MLFlowLogger), "Need MLFlowLogger"

    slides = get_slides(
        pd.read_csv(Path(config.dataset), index_col=0), Path(config.slides_folder)
    )
    output_path = Path(config.output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    process_items(
        slides,
        process_item=make_remote_process_slide(
            level=config.level, output_path=output_path
        ),
    )

    logger.log_artifacts(str(output_path), "tissue_masks")


if __name__ == "__main__":
    main()
