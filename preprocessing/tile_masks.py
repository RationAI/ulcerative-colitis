from collections.abc import Iterable
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import cast

import hydra
import mlflow.artifacts
import numpy as np
import pandas as pd
import pyvips
import ray
from omegaconf import DictConfig
from rationai.masks import process_items, tile_mask, write_big_tiff
from rationai.mlkit import autolog, with_cli_args
from rationai.mlkit.lightning.loggers import MLFlowLogger


ray.init(runtime_env={"excludes": [".git", ".venv"]})


def download_slide_tiles(uris: Iterable[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    slidess, tiless = [], []
    for uri in uris:
        path = mlflow.artifacts.download_artifacts(artifact_uri=uri)
        slidess.append(pd.read_parquet(Path(path) / "slides.parquet"))
        tiless.append(pd.read_parquet(Path(path) / "tiles.parquet"))

    slides = pd.concat(slidess).reset_index(drop=True)
    tiles = pd.concat(tiless).reset_index(drop=True)
    return slides, tiles


@ray.remote(memory=4 * 1024**3)
def process_slide(
    slide: pd.Series,
    tiles: pd.DataFrame,  # tiles are automatically serialized by Ray
    output_folder: Path,
) -> None:
    slide_tiles = tiles[tiles["slide_id"] == slide.id]

    blur_slide_tiles = slide_tiles[slide_tiles["blur"] > 0.25]
    artifacts_slide_tiles = slide_tiles[slide_tiles["artifacts"] > 0.25]
    slide_tiles = slide_tiles.drop(blur_slide_tiles.index)
    slide_tiles = slide_tiles.drop(artifacts_slide_tiles.index)

    for folder, tiles_subset in [
        ("blur_tiles", blur_slide_tiles),
        ("artifacts_tiles", artifacts_slide_tiles),
        ("clean_tiles", slide_tiles),
    ]:
        if tiles_subset.empty:
            continue

        mask = tile_mask(
            tiles_subset,
            tile_extent=(slide["tile_extent_x"], slide["tile_extent_y"]),
            size=(slide["extent_x"], slide["extent_y"]),
        )
        mask = cast("pyvips.Image", pyvips.Image.new_from_array(np.array(mask)))

        mask_path = output_folder / folder / f"{Path(slide['path']).stem}.tiff"
        mask_path.parent.mkdir(parents=True, exist_ok=True)
        write_big_tiff(
            mask,
            mask_path,
            mpp_x=slide["mpp_x"],
            mpp_y=slide["mpp_y"],
        )


@with_cli_args(["+preprocessing=tile_masks"])
@hydra.main(config_path="../configs", config_name="preprocessing", version_base=None)
@autolog
def main(config: DictConfig, logger: MLFlowLogger) -> None:
    slides, tiles = download_slide_tiles(config.dataset.uris.values())
    tiles_ref = ray.put(tiles)

    with TemporaryDirectory() as output_dir:
        process_items(
            (slide for _, slide in slides.iterrows()),
            process_item=process_slide,
            fn_kwargs={
                "tiles": tiles_ref,
                "output_folder": Path(output_dir),
            },
            max_concurrent=config.max_concurrent,
        )
        logger.log_artifacts(local_dir=output_dir, artifact_path=config.artifact_path)


if __name__ == "__main__":
    main()
