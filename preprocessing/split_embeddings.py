from pathlib import Path
from typing import cast

import mlflow
import mlflow.artifacts
import pandas as pd
import torch

from preprocessing.paths import EMBEDDING_REGIONS_PATH, EMBEDDINGS_PATH


def download_slides_tiles_dfs(uri: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Download slides and tiles DataFrames from MLflow artifacts."""
    artifacts = Path(mlflow.artifacts.download_artifacts(uri))
    return (
        pd.read_parquet(artifacts / "slides.parquet"),
        pd.read_parquet(artifacts / "tiles.parquet"),
    )


def split_embeddings(path: Path, slide_tiles: pd.DataFrame) -> None:
    """Split embeddings into individual slide files.

    Args:
        path (Path): Path to the .h5 file containing embeddings.
        slide_tiles (pd.DataFrame): DataFrame containing slide tile information.
    """
    embeddings = cast("torch.Tensor", torch.load(path, map_location="cpu"))

    slide_tiles = slide_tiles.reset_index(drop=True)
    for region in slide_tiles["region"].unique():
        region_tiles = slide_tiles.query(f"region == {region}")
        region_embeddings = embeddings[region_tiles.index.to_numpy()]

        torch.save(
            region_embeddings,
            EMBEDDING_REGIONS_PATH / f"{path.stem}_region_{region:02d}.pt",
        )


def main() -> None:
    uri = "mlflow-artifacts:/86/0f605c9479574c8498f64ffea5f87508/artifacts/"
    for split in ("train", "test preliminary", "test final"):
        slides, tiles = download_slides_tiles_dfs(uri)
        (EMBEDDING_REGIONS_PATH / split).mkdir(parents=True, exist_ok=True)

        for _, slide_metadata in slides.iterrows():
            slide_tiles = tiles.query(f"slide_id == {slide_metadata['id']!s}")
            slide_name = Path(slide_metadata["path"]).stem
            split_embeddings(
                (EMBEDDINGS_PATH / split / slide_name).with_suffix(".pt"), slide_tiles
            )

    mlflow.set_experiment(experiment_name="Ulcerative Colitis")
    with mlflow.start_run(run_name="📂 Dataset: Embedding Regions"):
        mlflow.log_artifact(str(EMBEDDING_REGIONS_PATH))


if __name__ == "__main__":
    main()
