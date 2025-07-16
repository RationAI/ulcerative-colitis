from pathlib import Path
from typing import cast

import h5py
import mlflow
import torch

from preprocessing.paths import EMBEDDINGS_PATH


def convert_embeddings(path: Path) -> None:
    """Convert embeddings from .pt to .h5 format.

    Args:
        path (Path): Path to the .pt file containing embeddings.
    """
    slide_embeddings = cast("torch.Tensor", torch.load(path, map_location="cpu"))

    with h5py.File(path.with_suffix(".h5"), "w") as f:
        f.create_dataset("embeddings", data=slide_embeddings.numpy())


def main() -> None:
    for embeddings_file in EMBEDDINGS_PATH.rglob("*.pt"):
        convert_embeddings(embeddings_file)

    mlflow.set_experiment(experiment_name="Ulcerative Colitis")
    with mlflow.start_run(run_name="📂 Dataset: Embeddings (H5)"):
        for embeddings_file in EMBEDDINGS_PATH.rglob("*.h5"):
            mlflow.log_artifact(
                str(embeddings_file),
                artifact_path=str(embeddings_file.relative_to(EMBEDDINGS_PATH)),
            )


if __name__ == "__main__":
    main()
