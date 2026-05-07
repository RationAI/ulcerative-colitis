from enum import Enum

import hydra
import mlflow
import numpy as np
import pandas as pd
from omegaconf import DictConfig
from rationai.mlkit import autolog, with_cli_args
from rationai.mlkit.lightning.loggers import MLFlowLogger

from postprocessing.utils import load_folds, load_label_map


class Confidence(Enum):
    ENTROPY = "entropy"
    HERFINDAHL = "herfindahl"
    STD = "std"


def absorption_distribution(
    neut_df: pd.DataFrame, nlow_df: pd.DataFrame, nhigh_df: pd.DataFrame
) -> np.ndarray:
    low = np.array(nlow_df["prediction"].tolist())   # (n, 3)
    high = np.array(nhigh_df["prediction"].tolist())  # (n, 4)

    a = 1.0 - np.array(neut_df["prediction"])  # P(neut -> nancy_low)
    b = low[:, 2]   # P(nancy_low  -> nancy_high)
    c = low[:, 0]   # P(nancy_low  -> NHI-0)
    d = high[:, 0]  # P(nancy_high -> nancy_low)
    e = high[:, 1]  # P(nancy_high -> NHI-2)
    f = high[:, 2]  # P(nancy_high -> NHI-3)

    denom = 1.0 - b * d
    flow_low = a + d - a * d          # absorption flow factor for NHI 0/1
    flow_high = 1.0 - a * (1.0 - b)  # absorption flow factor for NHI 2/3/4

    return np.column_stack(
        [
            c * flow_low / denom,             # P(NHI-0)
            low[:, 1] * flow_low / denom,     # P(NHI-1)
            e * flow_high / denom,            # P(NHI-2)
            f * flow_high / denom,            # P(NHI-3)
            high[:, 3] * flow_high / denom,   # P(NHI-4)
        ]
    )


def compute_confidence(pi: np.ndarray, mode: Confidence) -> np.ndarray:
    match mode:
        case Confidence.ENTROPY:
            safe = np.where(pi > 0, pi, 1.0)
            return 1.0 - (-(pi * np.log(safe) / np.log(5)).sum(axis=1))
        case Confidence.HERFINDAHL:
            return (np.square(pi).sum(axis=1) - 0.2) / 0.8
        case Confidence.STD:
            classes = np.array([0, 1, 2, 3, 4], dtype=float)
            mu = (pi * classes).sum(axis=1, keepdims=True)
            var = (pi * (classes - mu) ** 2).sum(axis=1)
            return np.maximum(0.0, 1.0 - np.sqrt(var))


@with_cli_args(["+postprocessing=markov_chain_confidence"])
@hydra.main(config_path="../configs", config_name="postprocessing", version_base=None)
@autolog
def main(config: DictConfig, logger: MLFlowLogger) -> None:
    confidence_mode = Confidence(config.confidence)
    label_map = load_label_map(config.dataset.mlflow_uris.dataset)
    folds: list[str] = list(config.folds)
    folds_data = load_folds(config.predictions.mlflow_uris, folds, label_map)

    rows = []
    for fold, fold_data in zip(folds, folds_data, strict=True):
        pi = absorption_distribution(
            fold_data["neutrophils"], fold_data["nancy_low"], fold_data["nancy_high"]
        )
        confidence = compute_confidence(pi, confidence_mode)

        for slide, pi_row, conf in zip(
            fold_data["neutrophils"].index, pi, confidence, strict=True
        ):
            rows.append(
                {
                    "slide": slide,
                    "fold": fold,
                    "pi_0": pi_row[0],
                    "pi_1": pi_row[1],
                    "pi_2": pi_row[2],
                    "pi_3": pi_row[3],
                    "pi_4": pi_row[4],
                    "confidence": conf,
                }
            )

    mlflow.log_table(pd.DataFrame(rows), artifact_file="confidence.json")


if __name__ == "__main__":
    main()
