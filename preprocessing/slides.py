from pathlib import Path

import pandas as pd


def _get_slides_ikem(df: pd.DataFrame, slides_folder: Path) -> list[Path]:
    slide_paths = []
    for case_id in df.index:
        # IKEM has only case level labels, so we need to find all slides for the case
        for slide_path in slides_folder.glob(f"{case_id.replace('/', '_')}*.czi"):
            slide_paths.append(slide_path)
    return slide_paths


def _get_slides_ftn(df: pd.DataFrame, slides_folder: Path) -> list[Path]:
    slide_paths = []
    for slide_id in df.index:
        slide_path = slides_folder / f"{slide_id.replace('/', '_')}.czi"
        assert slide_path.exists(), f"Slide {slide_path} does not exist"
        slide_paths.append(slide_path)
    return slide_paths


def _get_slides_knl_patos(df: pd.DataFrame, slides_folder: Path) -> list[Path]:
    slide_paths = []
    for slide_id in df.index:
        slide_path = slides_folder / f"{slide_id}.czi"
        assert slide_path.exists(), f"Slide {slide_path} does not exist"
        slide_paths.append(slide_path)
    return slide_paths


def get_slides(df: pd.DataFrame, slides_folder: Path, cohort: str) -> list[Path]:
    match cohort:
        case "ikem":
            return _get_slides_ikem(df, slides_folder)
        case "ftn":
            return _get_slides_ftn(df, slides_folder)
        case "knl_patos":
            return _get_slides_knl_patos(df, slides_folder)
        case _:
            raise ValueError(f"Unknown cohort: {cohort}")
