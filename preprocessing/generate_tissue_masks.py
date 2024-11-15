from pathlib import Path

import pyvips
import ray
from openslide import PROPERTY_NAME_MPP_X, PROPERTY_NAME_MPP_Y, OpenSlide
from rationai.masks import (
    process_items,
    tissue_mask,
    write_big_tiff,
)


BASE_FOLDER = Path("/mnt/data/Projects/inflammatory_bowel_dissease/ulcerative_colitis/")
SLIDES_PATH = BASE_FOLDER / "tiff"
TISSUE_MASKS_PATH = BASE_FOLDER / "tissue_masks"
LEVEL = 3


@ray.remote
def process_slide(slide_path: Path) -> None:
    with OpenSlide(slide_path) as slide:
        downsample = slide.level_downsamples[LEVEL]
        xres = float(slide.properties[PROPERTY_NAME_MPP_X]) * downsample
        yres = float(slide.properties[PROPERTY_NAME_MPP_Y]) * downsample

    slide = pyvips.Image.new_from_file(slide_path, page=LEVEL)
    mask = tissue_mask(slide)
    mask_path = TISSUE_MASKS_PATH / slide_path.with_suffix(".tiff").name
    mask_path.parent.mkdir(exist_ok=True, parents=True)
    write_big_tiff(mask, path=mask_path, xres=xres, yres=yres)


def main() -> None:
    slides = SLIDES_PATH.rglob("*.tiff")
    process_items(list(slides), process_item=process_slide)


if __name__ == "__main__":
    main()
