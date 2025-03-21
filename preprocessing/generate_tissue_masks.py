from pathlib import Path
from typing import cast

import mlflow
import pyvips
import ray
from openslide import OpenSlide
from rationai.masks import (
    process_items,
    slide_resolution,
    tissue_mask,
    write_big_tiff,
)


BASE_FOLDER = Path("/mnt/data/Projects/inflammatory_bowel_dissease/ulcerative_colitis/")
SLIDES_PATH = BASE_FOLDER / "data_tiff" / "20x"
TISSUE_MASKS_PATH = BASE_FOLDER / "tissue_masks" / "20x"
LEVEL = 3


@ray.remote
def process_slide(slide_path: Path) -> None:
    with OpenSlide(slide_path) as slide:
        mpp_x, mpp_y = slide_resolution(slide, level=LEVEL)

    slide = cast("pyvips.Image", pyvips.Image.new_from_file(slide_path, page=LEVEL))
    mask = tissue_mask(slide, mpp=(mpp_x + mpp_y) / 2)
    mask_path = TISSUE_MASKS_PATH / slide_path.with_suffix(".tiff").name
    mask_path.parent.mkdir(exist_ok=True, parents=True)
    write_big_tiff(mask, path=mask_path, mpp_x=mpp_x, mpp_y=mpp_y)


def main() -> None:
    slides = SLIDES_PATH.rglob("*.tiff")
    process_items(list(slides), process_item=process_slide)

    mlflow.set_experiment(experiment_name="IKEM")
    with mlflow.start_run(run_name="📂 Tissue Masks: Ulcerative Colitis"):
        mlflow.set_tag("mlflow.user", "AdamK")
        mlflow.log_artifacts(str(TISSUE_MASKS_PATH))


if __name__ == "__main__":
    main()
