from pathlib import Path
from tempfile import TemporaryDirectory

import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig
from rationai.mlkit import autolog, with_cli_args
from rationai.mlkit.lightning.loggers import MLFlowLogger
from sklearn.metrics import confusion_matrix

from postprocessing.ensembling import run_ensembling
from postprocessing.utils import BINARY_THRESHOLD, load_maps, load_predictions


METHODS = ["ensembling", "hierarchical"]


def labeled_confusion_matrix(
    y_true: np.ndarray, y_pred: np.ndarray, labels: list[int]
) -> pd.DataFrame:
    # rows are true labels, columns predicted labels
    return pd.DataFrame(
        confusion_matrix(y_true, y_pred, labels=labels),
        index=pd.Index([f"true_{label}" for label in labels]),
        columns=pd.Index([f"pred_{label}" for label in labels]),
    )


@with_cli_args(["+postprocessing=confusion_matrix"])
@hydra.main(config_path="../configs", config_name="postprocessing", version_base=None)
@autolog
def main(config: DictConfig, logger: MLFlowLogger) -> None:
    label_map, _ = load_maps(config)
    data = load_predictions(config.predictions.mlflow_uris, label_map)
    _, _, results = run_ensembling(data)

    y_true = results["nancy"].to_numpy()
    with TemporaryDirectory() as tmpdir:
        for method in METHODS:
            y_pred = results[f"pred_{method}"].to_numpy()
            matrices = {
                method: labeled_confusion_matrix(y_true, y_pred, [0, 1, 2, 3, 4]),
                f"{method}_binary": labeled_confusion_matrix(
                    (y_true >= BINARY_THRESHOLD).astype(int),
                    (y_pred >= BINARY_THRESHOLD).astype(int),
                    [0, 1],
                ),
            }
            for name, matrix in matrices.items():
                matrix.to_csv(Path(tmpdir) / f"{name}.csv")
                # row-normalized: diagonal is the per-class recall
                matrix.div(matrix.sum(axis=1).replace(0, 1), axis=0).to_csv(
                    Path(tmpdir) / f"{name}_normalized.csv"
                )
        logger.log_artifacts(tmpdir, artifact_path="confusion_matrices")


if __name__ == "__main__":
    main()
