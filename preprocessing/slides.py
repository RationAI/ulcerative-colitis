from pathlib import Path

import pandas as pd


def get_slides(df: pd.DataFrame, slides_folder: Path) -> list[Path]:
    slide_paths = []
    for slide_id in df.index:
        slide_path = slides_folder / f"{slide_id}.czi"
        assert slide_path.exists(), f"Slide {slide_path} does not exist"
        slide_paths.append(slide_path)
    return slide_paths
