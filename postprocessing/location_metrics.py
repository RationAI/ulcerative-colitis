from pathlib import Path
from tempfile import TemporaryDirectory

import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig
from rationai.mlkit import autolog, with_cli_args
from rationai.mlkit.lightning.loggers import MLFlowLogger
from sklearn.metrics import accuracy_score, cohen_kappa_score

from postprocessing.bootstrap import case_bootstrap
from postprocessing.ensembling import run_ensembling
from postprocessing.utils import BINARY_THRESHOLD, load_maps, load_predictions
from preprocessing.create_dataset import get_labels


METHODS = ["ensembling", "hierarchical"]

# reported locations are cekum, transverzum, descendens and rektum
LOCATION_MAPPING = {
    "cekoascendens": "cekum",
    "rektosigma": "rektum",
    # typos in the IKEM label files
    "retum": "rektum",
    "rektun": "rektum",
}
# location missing in the label files
DEFAULT_LOCATION = "rektum"


def load_locations(folder: str, labels: list[str]) -> dict[str, str]:
    # location is only in the raw IKEM label files, one per case
    locations = get_labels(Path(folder), labels)["lokalita"]
    locations = locations.str.strip().str.lower().replace(LOCATION_MAPPING)
    return locations.fillna(DEFAULT_LOCATION).to_dict()


def compute_location_metrics(
    y_true: np.ndarray, y_pred: np.ndarray
) -> dict[str, float]:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        # off by at most one grade counts as correct
        "accuracy_within_1": float(np.mean(np.abs(y_true - y_pred) <= 1)),
        "binary_accuracy": accuracy_score(
            y_true >= BINARY_THRESHOLD, y_pred >= BINARY_THRESHOLD
        ),
        "cohen_kappa_quadratic": cohen_kappa_score(y_true, y_pred, weights="quadratic"),
    }


@with_cli_args(["+postprocessing=location_metrics"])
@hydra.main(config_path="../configs", config_name="postprocessing", version_base=None)
@autolog
def main(config: DictConfig, logger: MLFlowLogger) -> None:
    label_map, case_map = load_maps(config)
    data = load_predictions(config.predictions.mlflow_uris, label_map)
    _, _, results = run_ensembling(data)

    locations = load_locations(config.dataset.folder, config.dataset.labels)
    results["case"] = results["slide"].map(case_map)
    results["location"] = (
        results["case"].str.split("/").str[-1].map(locations).fillna(DEFAULT_LOCATION)
    )

    strata = {"all": results} | dict(list(results.groupby("location")))

    rows = []
    for method in METHODS:
        for location, stratum in strata.items():
            table = case_bootstrap(
                stratum["nancy"].to_numpy(),
                stratum[f"pred_{method}"].to_numpy(),
                stratum["case"].to_numpy(),
                compute_location_metrics,
                config.n_resamples,
                config.confidence_level,
                np.random.default_rng(config.seed),
            )
            n_slides, n_cases = len(stratum), stratum["case"].nunique()
            logger.log_metrics(
                {
                    f"{method}/{location}/n_slides": n_slides,
                    f"{method}/{location}/n_cases": n_cases,
                }
                | {
                    f"{method}/{location}/{metric}{suffix}": float(row[column])
                    for metric, row in table.iterrows()
                    for column, suffix in [
                        ("estimate", ""),
                        ("ci_lower", "_ci_lower"),
                        ("ci_upper", "_ci_upper"),
                    ]
                }
            )
            rows.append(
                table.rename_axis("metric")
                .reset_index()
                .assign(
                    method=method,
                    location=location,
                    n_slides=n_slides,
                    n_cases=n_cases,
                )
            )

    with TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "location_metrics.csv"
        pd.concat(rows)[
            [
                "method",
                "location",
                "n_slides",
                "n_cases",
                "metric",
                "estimate",
                "ci_lower",
                "ci_upper",
            ]
        ].to_csv(output_path, index=False)
        logger.log_artifact(str(output_path))


if __name__ == "__main__":
    main()
