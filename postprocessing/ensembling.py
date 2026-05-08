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

from postprocessing.utils import load_predictions, load_label_map


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


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "specificity": macro_specificity(y_true, y_pred),
        "cohen_kappa": cohen_kappa_score(y_true, y_pred),
    }


PredictionData = dict[str, pd.DataFrame]


def run_ensembling(
    data: PredictionData,
) -> tuple[dict[str, float], dict[str, float], pd.DataFrame]:
    neut_df = data["neutrophils"]
    nlow_df = data["nancy_low"]
    nhigh_df = data["nancy_high"]

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
    ens_metrics = compute_metrics(y_true, ens_pred)

    # Hierarchical: neutrophils task alone routes
    hier_pred = np.where(neut_prob >= 0.5, high_branch, low_branch)
    hier_metrics = compute_metrics(y_true, hier_pred)

    results = [
        {
            "slide": slide,
            "nancy": int(yt),
            "pred_ensembling": int(ens_yp),
            "pred_hierarchical": int(hier_yp),
        }
        for slide, yt, ens_yp, hier_yp in zip(
            neut_df.index, y_true, ens_pred, hier_pred, strict=True
        )
    ]

    return ens_metrics, hier_metrics, pd.DataFrame(results)


@with_cli_args(["+postprocessing=ensembling"])
@hydra.main(config_path="../configs", config_name="postprocessing", version_base=None)
@autolog
def main(config: DictConfig, logger: MLFlowLogger) -> None:
    label_map = load_label_map(config.dataset.mlflow_uris.dataset)
    uris = {task: config.predictions.mlflow_uris[task] for task in ["neutrophils", "nancy_low", "nancy_high"]}
    data = load_predictions(uris, label_map)
    ens_metrics, hier_metrics, results = run_ensembling(data)

    logger.log_metrics({f"ensembling/{k}": v for k, v in ens_metrics.items()})
    logger.log_metrics({f"hierarchical/{k}": v for k, v in hier_metrics.items()})

    with TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "results.csv"
        results.to_csv(output_path, index=False)
        logger.log_artifact(str(output_path))


if __name__ == "__main__":
    main()
