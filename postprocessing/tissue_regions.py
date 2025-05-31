from pathlib import Path

import mlflow
import mlflow.artifacts
import pandas as pd
import torch
from rationai.masks.mask_builders import ScalarMaskBuilder

from preprocessing.tiling import add_regions


SLIDES = ("1468_23_HE", "5005_23_HE")


def main() -> None:
    folder = Path(
        mlflow.artifacts.download_artifacts(
            "mlflow-artifacts:/27/12b7bd13ec474a5c889a4642e3c951bd/artifacts/Ulcerative Colitis - test1"
        )
    )

    slides_df = pd.read_parquet(folder / "slides.parquet")
    tiles_df = pd.read_parquet(folder / "tiles.parquet")
    slides_df["slide_name"] = slides_df["path"].apply(
        lambda x: x.split("/")[-1].split(".")[0]
    )

    slide_ids = slides_df.query("slide_name in @SLIDES")["id"].tolist()
    slides_df = slides_df.query("id in @slide_ids").reset_index(drop=True)
    tiles_df = tiles_df.query("slide_id in @slide_ids").reset_index(drop=True)

    tiles_df = add_regions(slides_df, tiles_df)
    n_regions = tiles_df["tissue_region"].nunique() + 1  # +1 for background
    tiles_df["tissue_region"] = tiles_df["tissue_region"] + 1

    for slide_id in slide_ids:
        slide = slides_df.query(f"id == {slide_id}").iloc[0]
        mask_builder = ScalarMaskBuilder(
            save_dir=Path("masks/tissue_regions"),
            filename=slide.slide_name,
            extent_x=slide.extent_x,
            extent_y=slide.extent_y,
            mpp_x=slide.mpp_x,
            mpp_y=slide.mpp_y,
            extent_tile=slide.tile_extent_x,
            stride=slide.stride_x,
        )

        tiles = tiles_df.query(f"slide_id == {slide_id}")
        mask_builder.update(
            torch.tensor(tiles["x"].to_numpy()),
            torch.tensor(tiles["y"].to_numpy()),
            torch.tensor(tiles["tissue_region"]) / n_regions,
        )

        mlflow.log_artifact(
            str(mask_builder.save()), artifact_path=str(mask_builder.save_dir)
        )


if __name__ == "__main__":
    mlflow.set_experiment("IKEM")
    with mlflow.start_run(run_name="🗣️ Explaining: Ulcerative Colitis - Tissue Regions"):
        main()
