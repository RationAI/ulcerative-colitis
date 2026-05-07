from pathlib import Path
from tempfile import TemporaryDirectory

import hydra
import mlflow
import numpy as np
import pandas as pd
from omegaconf import DictConfig
from rationai.mlkit import autolog, with_cli_args
from rationai.mlkit.lightning.loggers import MLFlowLogger
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    precision_score,
    recall_score,
)

from postprocessing.utils import load_folds, load_label_map


METRIC_KEYS = ["accuracy", "precision", "recall", "specificity", "cohen_kappa"]


def macro_specificity(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    cm = confusion_matrix(y_true, y_pred)
    total = cm.sum()
    specs = []
    for i in range(len(cm)):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        tn = total - tp - fp - fn
        specs.append(tn / (tn + fp) if (tn + fp) > 0 else 0.0)
    return float(np.mean(specs))


def fold_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "specificity": macro_specificity(y_true, y_pred),
        "cohen_kappa": cohen_kappa_score(y_true, y_pred),
    }


def log_table_metrics(
    label: str, folds: list[str], per_fold: list[dict[str, float]]
) -> None:
    for fold, m in zip(folds, per_fold, strict=True):
        fold_key = fold.replace("/", "_")
        mlflow.log_metrics({f"{label}/fold_{fold_key}/{k}": v for k, v in m.items()})
    means = {k: float(np.mean([m[k] for m in per_fold])) for k in METRIC_KEYS}
    mlflow.log_metrics({f"{label}/mean/{k}": v for k, v in means.items()})


FoldData = dict[str, pd.DataFrame]


def run_ensembling(
    folds: list[str], folds_data: list[FoldData]
) -> tuple[list[dict[str, float]], list[dict[str, float]], pd.DataFrame]:
    ens_per_fold: list[dict[str, float]] = []
    hier_per_fold: list[dict[str, float]] = []
    results = []

    for fold, fold_data in zip(folds, folds_data, strict=True):
        neut_df = fold_data["neutrophils"]
        nlow_df = fold_data["nancy_low"]
        nhigh_df = fold_data["nancy_high"]

        neut_prob = np.array(neut_df["prediction"])
        low_probs = np.array(nlow_df["prediction"].tolist())
        high_probs = np.array(nhigh_df["prediction"].tolist())

        y_true = neut_df["nancy"].values
        low_branch = low_probs[:, :2].argmax(axis=1)
        high_branch = high_probs[:, 1:].argmax(axis=1) + 2

        # Ensembling: soft majority vote across three tasks
        route_high_ens = (
            neut_prob + low_probs[:, 2] + high_probs[:, 1:].sum(axis=1)
        ) >= 1.5
        ens_pred = np.where(route_high_ens, high_branch, low_branch)
        ens_per_fold.append(fold_metrics(y_true, ens_pred))

        # Hierarchical: neutrophils task alone routes
        hier_pred = np.where(neut_prob >= 0.5, high_branch, low_branch)
        hier_per_fold.append(fold_metrics(y_true, hier_pred))

        for slide, yt, ens_yp, hier_yp in zip(
            neut_df.index, y_true, ens_pred, hier_pred, strict=True
        ):
            results.append(
                {
                    "fold": fold,
                    "slide": slide,
                    "nancy": int(yt),
                    "pred_ensembling": int(ens_yp),
                    "pred_hierarchical": int(hier_yp),
                }
            )

    return ens_per_fold, hier_per_fold, pd.DataFrame(results)


@with_cli_args(["+postprocessing=ensembling"])
@hydra.main(config_path="../configs", config_name="postprocessing", version_base=None)
@autolog
def main(config: DictConfig, logger: MLFlowLogger) -> None:
    label_map = load_label_map(config.dataset.mlflow_uris.dataset)
    folds: list[str] = list(config.folds)
    folds_data = load_folds(config.predictions.mlflow_uris, folds, label_map)
    ens_per_fold, hier_per_fold, results = run_ensembling(folds, folds_data)

    log_table_metrics("ensembling", folds, ens_per_fold)
    log_table_metrics("hierarchical", folds, hier_per_fold)

    with TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "results.csv"
        results.to_csv(output_path, index=False)
        logger.log_artifact(str(output_path))


if __name__ == "__main__":
    main()
