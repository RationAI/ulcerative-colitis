from functools import partial
from pathlib import Path

import hydra
import mlflow.artifacts
import numpy as np
import pandas as pd
import pyvips
from omegaconf import DictConfig, ListConfig
from rationai.mlkit import with_cli_args
from rationai.mlkit.autolog import autolog
from rationai.mlkit.lightning.loggers import MLFlowLogger
from sklearn.metrics import (
    cohen_kappa_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)


def add_path(path: str, folder: Path, suffix: str) -> Path:
    return folder / Path(path).with_suffix(suffix).name


def download_data(datasets: ListConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_dfs, test_dfs = [], []
    for dataset in datasets:
        neutrophils = Path(mlflow.artifacts.download_artifacts(dataset.neutrophils_uri))
        tissue_masks = Path(
            mlflow.artifacts.download_artifacts(dataset.tissue_mask_uri)
        )

        add_tissue_mask = partial(add_path, folder=tissue_masks, suffix=".tiff")
        add_neutrophils = partial(add_path, folder=neutrophils, suffix=".parquet")
        if "train" in dataset.tiling_uris:
            train_dfs.append(
                pd.read_parquet(
                    Path(mlflow.artifacts.download_artifacts(dataset.tiling_uris.train))
                    / "slides.parquet",
                )
                .set_index("id")
                .assign(tissue_mask=lambda df, f=add_tissue_mask: df["path"].apply(f))  # type: ignore[reportCallIssues]
                .assign(neutrophils=lambda df, f=add_neutrophils: df["path"].apply(f))  # type: ignore[reportCallIssues]
            )

        if "test_preliminary" in dataset.tiling_uris:
            test_dfs.append(
                pd.read_parquet(
                    Path(
                        mlflow.artifacts.download_artifacts(
                            dataset.tiling_uris.test_preliminary
                        )
                    )
                    / "slides.parquet",
                )
                .set_index("id")
                .assign(tissue_mask=lambda df, f=add_tissue_mask: df["path"].apply(f))  # type: ignore[reportCallIssues]
                .assign(neutrophils=lambda df, f=add_neutrophils: df["path"].apply(f))  # type: ignore[reportCallIssues]
            )

    return pd.concat(train_dfs), pd.concat(test_dfs)


def get_tissue_size(tissue_mask_path: Path) -> float:
    """Calculate tissue size in mm^2."""
    image = pyvips.Image.new_from_file(tissue_mask_path)
    mpp_x = 1000 / image.get("Xres")  # type: ignore[pyvips]
    mpp_y = 1000 / image.get("Yres")  # type: ignore[pyvips]
    return image.avg() * image.width * image.height / 255 * mpp_x * mpp_y * 1e-6  # type: ignore[pyvips]


def create_dataset(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    x, y = [], []
    for _, row in df.iterrows():
        expected_num_neutrophils = pd.read_parquet(row["neutrophils"])[
            "probability"
        ].sum()
        tissue_size = get_tissue_size(row["tissue_mask"])

        # expected number of neutrophils per mm^2
        x.append(expected_num_neutrophils / tissue_size)
        y.append(row["nancy_index"] >= 2)

    return np.array(x, dtype=np.float32), np.array(y, dtype=np.int8)


def train(x_train: np.ndarray, y_train: np.ndarray) -> float:
    fpr, tpr, thresholds = roc_curve(y_train, x_train)
    accuracy = (tpr * np.sum(y_train) + (1 - fpr) * np.sum(1 - y_train)) / len(y_train)
    return thresholds[np.argmax(accuracy)]


def test(
    x_test: np.ndarray,
    y_test: np.ndarray,
    test_df: pd.DataFrame,
    threshold: float,
    logger: MLFlowLogger,
) -> None:
    y_pred = x_test >= threshold

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    logger.log_metrics(
        {
            "test/accuracy": (tp + tn) / (tp + tn + fp + fn),
            "test/AUC": float(roc_auc_score(y_test, x_test)),
            "test/precision": (tp / (tp + fp)) if tp + fp > 0 else 0.0,
            "test/recall": (tp / (tp + fn)) if tp + fn > 0 else 0.0,
            "test/specificity": (tn / (tn + fp)) if tn + fp > 0 else 0.0,
            "test/kappa": float(cohen_kappa_score(y_test, y_pred)),
            "threshold": threshold,
        }
    )

    logger.log_table(
        {
            "slide": test_df["path"].apply(lambda p: p.stem),
            "prediction": x_test.tolist(),
            "prediction_class": y_pred.tolist(),
        },
        artifact_file="predictions.json",
    )


@with_cli_args(["+ml=neutrophils"])
@hydra.main(config_path="../configs", config_name="ml", version_base=None)
@autolog
def main(config: DictConfig, logger: MLFlowLogger) -> None:
    train_df, test_df = download_data(
        config.dataset.get("datasets", {"": config.dataset}).values()
    )

    x_train, y_train = create_dataset(train_df)
    x_test, y_test = create_dataset(test_df)

    threshold = train(x_train, y_train)
    test(x_test, y_test, test_df, threshold, logger)


if __name__ == "__main__":
    main()
