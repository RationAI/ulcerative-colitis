from collections.abc import Callable
from pathlib import Path
from tempfile import TemporaryDirectory

import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig
from rationai.mlkit import autolog, with_cli_args
from rationai.mlkit.lightning.loggers import MLFlowLogger
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    precision_score,
    recall_score,
)

from postprocessing.ensembling import compute_metrics, run_ensembling
from postprocessing.utils import BINARY_THRESHOLD, load_maps, load_predictions


METHODS = ["ensembling", "hierarchical"]

Metrics = Callable[[np.ndarray, np.ndarray], dict[str, float]]


def compute_binary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "specificity": recall_score(y_true, y_pred, pos_label=0, zero_division=0),
        "cohen_kappa": cohen_kappa_score(y_true, y_pred),
    }


def case_bootstrap(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    cases: np.ndarray,
    metrics: Metrics,
    n_resamples: int,
    confidence_level: float,
    rng: np.random.Generator,
) -> pd.DataFrame:
    # resample whole cases with replacement, slides of one case are correlated
    case_ids, case_index = np.unique(cases, return_inverse=True)
    slides_by_case = [np.flatnonzero(case_index == i) for i in range(len(case_ids))]

    samples = []
    for _ in range(n_resamples):
        drawn = rng.integers(len(case_ids), size=len(case_ids))
        idx = np.concatenate([slides_by_case[i] for i in drawn])
        samples.append(metrics(y_true[idx], y_pred[idx]))
    samples_df = pd.DataFrame(samples)

    alpha = (1.0 - confidence_level) / 2
    return pd.DataFrame(
        {
            "estimate": metrics(y_true, y_pred),
            "ci_lower": samples_df.quantile(alpha),
            "ci_upper": samples_df.quantile(1.0 - alpha),
        }
    )


@with_cli_args(["+postprocessing=bootstrap"])
@hydra.main(config_path="../configs", config_name="postprocessing", version_base=None)
@autolog
def main(config: DictConfig, logger: MLFlowLogger) -> None:
    label_map, case_map = load_maps(config)
    data = load_predictions(config.predictions.mlflow_uris, label_map)
    _, _, results = run_ensembling(data)

    y_true = results["nancy"].to_numpy()
    cases = results["slide"].map(case_map).to_numpy()

    # for both methods, NHI >= 2 coincides with their routing decision (neutrophils
    # model for hierarchical, soft vote of all tasks for ensembling)
    variants: list[tuple[str, Metrics, Callable[[np.ndarray], np.ndarray]]] = [
        ("", compute_metrics, lambda y: y),
        ("binary", compute_binary_metrics, lambda y: y >= BINARY_THRESHOLD),
    ]

    tables = []
    for method in METHODS:
        for variant, metrics, transform in variants:
            name = f"{method}_{variant}" if variant else method
            table = case_bootstrap(
                transform(y_true),
                transform(results[f"pred_{method}"].to_numpy()),
                cases,
                metrics,
                config.n_resamples,
                config.confidence_level,
                np.random.default_rng(config.seed),
            )
            logger.log_metrics(
                {
                    f"{name}/{metric}{suffix}": float(row[column])
                    for metric, row in table.iterrows()
                    for column, suffix in [
                        ("estimate", ""),
                        ("ci_lower", "_ci_lower"),
                        ("ci_upper", "_ci_upper"),
                    ]
                }
            )
            tables.append(table.rename_axis("metric").reset_index().assign(method=name))

    with TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "confidence_intervals.csv"
        pd.concat(tables)[
            ["method", "metric", "estimate", "ci_lower", "ci_upper"]
        ].to_csv(output_path, index=False)
        logger.log_artifact(str(output_path))


if __name__ == "__main__":
    main()
